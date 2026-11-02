#!/usr/bin/env python3
"""
Evaluation script for Task 3A.
Evaluates rosbag against the official eYRC Task 3A rubric (80 marks):
- 6 survivors: 3 critical, 3 stable
- Visit: 3 marks per survivor (6 x 3 = 18 marks max)
- Save: 5 marks per survivor (6 x 5 = 30 marks max)
- Hover at F6: 3 marks (hold at Z=13 for 1s after last stable survivor)
- Landing at F6: 3 marks (stay at F6 within +-0.4 on X and Y before 150s)
- Speed marks: 26 * (150 - T) / 150 * (S / 6)
Total: 80 marks
"""

import os
import sys
import sqlite3
import yaml
import json
import math
from rclpy.serialization import deserialize_message
from whycode_interfaces.msg import MarkerArray

TOLERANCE = 0.40  # +-0.4 whycode units

def in_sphere(pos, target, tol=TOLERANCE):
    dist = math.sqrt(
        (pos[0] - target[0])**2 +
        (pos[1] - target[1])**2 +
        (pos[2] - target[2])**2
    )
    return dist <= tol

def evaluate_bag(bag_dir, crit_names=None, stab_names=None, coords_map=None):
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    db_candidates = [
        os.path.join(bag_dir, "task_3a_0.db3"),
        os.path.join(bag_dir, "task_3a_0.mcap")
    ]
    db_path = None
    for cand in db_candidates:
        if os.path.exists(cand):
            db_path = cand
            break

    if not os.path.exists(meta_path) or db_path is None:
        print(f"Error: {bag_dir} is missing metadata.yaml or database file")
        return False, 0.0

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    duration_ns = meta["rosbag2_bagfile_information"]["duration"]["nanoseconds"]
    duration_s = duration_ns * 1e-9
    print("=" * 60)
    print(f"Task 3A Evaluation for bag: {bag_dir}")
    print(f"Bag Duration: {duration_s:.2f} s")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM topics")
    topics = dict(cursor.fetchall())
    id_by_name = {v: k for k, v in topics.items()}
    print(f"Topics recorded: {list(id_by_name.keys())}")

    if '/controller_heartbeat' not in id_by_name:
        print("WARNING: /controller_heartbeat missing in bag!")
    if '/whycode_node/markers' not in id_by_name:
        print("ERROR: /whycode_node/markers missing in bag!")
        return False, 0.0

    cursor.execute("SELECT MIN(timestamp) FROM messages")
    t0_raw = cursor.fetchone()[0]

    marker_topic_id = id_by_name['/whycode_node/markers']
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (marker_topic_id,))
    rows = cursor.fetchall()

    if not rows:
        print("ERROR: No messages in /whycode_node/markers!")
        return False, 0.0

    # Find t_takeoff (first time z < 14.6)
    t_takeoff = None
    for ts_raw, data in rows:
        msg = deserialize_message(data, MarkerArray)
        if msg.markers:
            z = msg.markers[0].position.position.z
            if z < 14.6:
                t_takeoff = ts_raw * 1e-9
                break

    if t_takeoff is None:
        t_takeoff = rows[0][0] * 1e-9

    print(f"Takeoff detected at t = 0 (offset {t_takeoff - rows[0][0]*1e-9:.2f}s from bag start)")

    # Read grid coordinates
    if coords_map is None:
        json_path = "src/swift_pico/src/grid_whycode_coordinates.json"
        if not os.path.exists(json_path):
            json_path = "/home/eyepatch/eYRC/eYRC_26-27_Khoj-o-Drone/src/swift_pico/src/grid_whycode_coordinates.json"
        with open(json_path) as f:
            coords_map = json.load(f)

    f6_pos = coords_map['F6']

    # Load detected survivors from service_3a.log if not provided
    if crit_names is None or stab_names is None:
        import re, ast
        candidate_logs = [os.path.join(bag_dir, "service_3a.log")]
        if bag_dir in ["task_3a", "."]:
            candidate_logs.append("service_3a.log")
        for candidate_log in candidate_logs:
            if os.path.exists(candidate_log):
                with open(candidate_log) as f:
                    content = f.read()
                    mc = re.search(r"Detected 3 Critical Survivors:\s*(\[[^\]]+\])", content)
                    ms = re.search(r"Detected 3 Stable Survivors:\s*(\[[^\]]+\])", content)
                    if mc and ms:
                        try:
                            crit_names = ast.literal_eval(mc.group(1))
                            stab_names = ast.literal_eval(ms.group(1))
                            break
                        except Exception:
                            pass
        if crit_names is None or stab_names is None:
            crit_names = ['B10', 'J9', 'I2']
            stab_names = ['B2', 'F10', 'J6']

    print(f"Target Survivors: Critical={crit_names}, Stable={stab_names}")
    visited_records = []
    # Every trajectory sample
    trajectory = []
    for ts_raw, data in rows:
        t = ts_raw * 1e-9 - t_takeoff
        msg = deserialize_message(data, MarkerArray)
        if msg.markers:
            p = msg.markers[0].position.position
            trajectory.append((t, p.x, p.y, p.z))

    print(f"Total trajectory points: {len(trajectory)}")

    # Check holds at all grid locations
    # A Visit is a hold of >= 1.0s at Z in [12.6, 13.4]
    # A Save is a hold of >= 2.0s at Z in [13.6, 14.4] right after Visit
    # Hover is a hold at F6 of >= 1.0s at Z=13
    # Landing is sitting at F6 with Z >= 14.8 until bag end

    # ── Official Portal Evaluation Logic ─────────────────────────────────
    # The official portal uses:
    # 1. Continuous Euclidean distance <= 0.40m from waypoint target:
    #    Visit: Z=13.0, continuous hold >= 1.00s
    #    Save:  Z=14.0, continuous hold >= 2.00s (must follow Visit)
    #    Any sample with dist > 0.40m resets the continuous hold timer to 0!
    # 2. Critical Survivors: Visit & Save must finish before the Switch.
    # 3. Stable Survivors: Visit must start AFTER the Switch.
    #    The Switch occurs when all 3 critical saves are complete, or at t = 55.00s.
    # 4. F6 Hover: continuous hold >= 1.00s at F6 (Z=13.0) after last stable save.
    # 5. F6 Landing: touchdown at F6 before 150.00s.
    # 6. Speed: 26 * (150 - T)/150 * (S / 6)

    def find_continuous_holds(tx, ty, tz, tol=0.40):
        """Finds all continuous intervals where Euclidean dist <= tol."""
        intervals = []
        cur_start = None
        last_t = None
        for t, x, y, z in trajectory:
            if t < 0:
                continue
            d = math.sqrt((x - tx)**2 + (y - ty)**2 + (z - tz)**2)
            if d <= tol:
                if cur_start is None:
                    cur_start = t
                last_t = t
            else:
                if cur_start is not None:
                    dur = last_t - cur_start
                    if dur >= 0.05:
                        intervals.append((cur_start, last_t, dur))
                    cur_start = None
        if cur_start is not None:
            dur = last_t - cur_start
            if dur >= 0.05:
                intervals.append((cur_start, last_t, dur))
        return intervals

    # Determine Switch time and evaluate survivors
    # Identify which survivors were critical vs stable by finding holds
    survivor_eval = {}
    for name, c in coords_map.items():
        if name == 'F6':
            continue
        tx, ty = c['x'], c['y']
        v_ints = find_continuous_holds(tx, ty, 13.0, tol=0.40)
        s_ints = find_continuous_holds(tx, ty, 14.0, tol=0.40)
        best_v = max([i[2] for i in v_ints], default=0.0)
        best_s = max([i[2] for i in s_ints], default=0.0)
        # Find earliest valid visit (dur >= 1.00s)
        earliest_v = next((i[1] for i in v_ints if i[2] >= 1.00), None)
        # Find earliest valid save (dur >= 2.00s) after a visit
        earliest_s = None
        if earliest_v is not None:
            earliest_s = next((i[1] for i in s_ints if i[2] >= 2.00 and i[0] >= earliest_v - 0.5), None)
        
        if earliest_v is not None or best_v >= 0.5:
            survivor_eval[name] = {
                'v_ints': v_ints, 's_ints': s_ints,
                'best_v': best_v, 'best_s': best_s,
                'v_time': earliest_v, 's_time': earliest_s
            }

    # Determine Switch time from critical saves
    crit_saved_times = []
    for name in crit_names:
        data = survivor_eval.get(name)
        if data and data['s_time'] is not None and data['s_time'] <= 55.0:
            crit_saved_times.append(data['s_time'])

    if len(crit_saved_times) == 3:
        switch_time = max(crit_saved_times)
    else:
        switch_time = 55.00

    print(f"\n--- Official Rubric Evaluation (Switch Time: {switch_time:.2f}s) ---")
    portal_visits = []
    portal_saves = []

    target_list = [(n, 'CRITICAL') for n in crit_names] + [(n, 'STABLE') for n in stab_names]
    for name, tag in target_list:
        data = survivor_eval.get(name, {
            'v_ints': [], 's_ints': [],
            'best_v': 0.0, 'best_s': 0.0,
            'v_time': None, 's_time': None
        })
        v_t = data['v_time']
        s_t = data['s_time']
        v_dur = data['best_v']
        s_dur = data['best_s']

        if tag == "CRITICAL":
            # Critical survivor: visit & save must be before switch
            v_ok = (v_t is not None and v_t <= switch_time)
            s_ok = (v_ok and s_t is not None and s_t <= switch_time)
        else:
            # Stable survivor: visit must START after switch
            v_valid_after_switch = any(i[2] >= 1.00 and i[0] >= switch_time - 0.05 for i in data['v_ints'])
            v_ok = v_valid_after_switch
            s_ok = (v_ok and s_t is not None and s_t > switch_time)

        status_v = f"Visited at {v_t:.2f}s (dur {v_dur:.2f}s)" if v_ok else f"Visit FAIL (best {v_dur:.2f}s / 1.00s)"
        status_s = f"Saved at {s_t:.2f}s (dur {s_dur:.2f}s)" if s_ok else f"Save FAIL (best {s_dur:.2f}s / 2.00s)"
        print(f"  [{tag}] {name}: {status_v} | {status_s}")

        if v_ok:
            portal_visits.append(name)
        if s_ok:
            portal_saves.append(name)

    # Check F6 hover: hold >= 1.00s at F6 Z=13 after last stable save
    last_save_t = max([survivor_eval[n]['s_time'] for n in portal_saves], default=switch_time)
    f6_v_ints = find_continuous_holds(f6_pos['x'], f6_pos['y'], 13.0, tol=0.40)
    hover_ok = any(i[2] >= 1.00 and i[0] >= last_save_t - 1.0 for i in f6_v_ints)
    hover_best = max([i[2] for i in f6_v_ints], default=0.0)

    # Check landing at F6 (must occur after last save / flight phase)
    landed = False
    landing_time = None
    f6_x, f6_y = f6_pos['x'], f6_pos['y']
    min_landing_t = max(40.0, last_save_t - 2.0)
    for p in reversed(trajectory):
        if p[0] < min_landing_t:
            break
        d_xy = math.hypot(p[1] - f6_x, p[2] - f6_y)
        if d_xy <= 0.40 and p[3] >= 14.50:
            landed = True
            landing_time = p[0]
        else:
            if landed:
                break

    T = landing_time if landed else duration_s
    v_marks = min(18.0, len(portal_visits) * 3.0)
    s_marks = min(30.0, len(portal_saves) * 5.0)
    h_marks = 3.0 if hover_ok else 0.0
    l_marks = 3.0 if (landed and len(portal_visits) >= 1) else 0.0
    speed_marks = 0.0
    if landed and T < 150.0:
        speed_marks = 26.0 * (150.0 - T) / 150.0 * (len(portal_saves) / 6.0)

    official_total = v_marks + s_marks + h_marks + l_marks + speed_marks

    print("\n" + "=" * 60)
    print("OFFICIAL PORTAL REPLICATED SCORE BREAKDOWN:")
    print(f"  Visits:  {v_marks:.2f} / 18.0 ({len(portal_visits)}/6 survivors: {portal_visits})")
    print(f"  Saves:   {s_marks:.2f} / 30.0 ({len(portal_saves)}/6 survivors: {portal_saves})")
    print(f"  Hover:   {h_marks:.2f} / 3.0 (Best: {hover_best:.2f}s)")
    print(f"  Landing: {l_marks:.2f} / 3.0 (Landed: {landed}, Touchdown T={T:.2f}s)")
    print(f"  Speed:   {speed_marks:.2f} / 26.0")
    print(f"  OFFICIAL SCORE: {official_total:.2f} / 80.0")
    print("=" * 60)

    return True, official_total

if __name__ == '__main__':
    bag = sys.argv[1] if len(sys.argv) > 1 else 'task_3a'
    evaluate_bag(bag)

