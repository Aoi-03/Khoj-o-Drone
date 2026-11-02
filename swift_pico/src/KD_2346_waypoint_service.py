#!/usr/bin/env python3
"""
Task 3A – Waypoint Service
==========================
1. Subscribes once to /image_raw.
2. Detects ArUco corner markers (80, 85, 90, 95) and warps arena to 900x900 px canvas.
3. Detects:
   - 3 Critical survivors (red triangles)
   - 3 Stable survivors (yellow circles)
   - 3 Black rectangular obstacles
4. Inflates obstacles to create off-limits collision boundaries.
5. Plans optimal critical route from F6 through the 3 critical survivors,
   routing around all obstacles using A* pathfinding.
6. Serves critical route via 'waypoints' (GetWaypoints service).
7. Serves stable route via 'replan_route' (ReplanRoute service), planning from
   the drone's current position at the Switch through the 3 stable survivors,
   ending at F6 for landing.
"""

import os
import sys
import time
import math
import heapq
import itertools
import json

# Ensure Fast-DDS shared memory configuration is applied if available
WS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FAST_DDS_CANDIDATES = [
    os.path.join(WS_DIR, 'install/swift_pico/share/swift_pico/config/fastdds_profile.xml'),
    os.path.join(WS_DIR, 'src/swift_pico/config/fastdds_profile.xml'),
]
for p in FAST_DDS_CANDIDATES:
    if os.path.exists(p):
        os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', p)
        break

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge

from waypoint_navigation.srv import GetWaypoints, ReplanRoute
from ament_index_python.packages import get_package_share_directory


# ──────────────────────────────────────────────────────────────────────────────
# Arena & Grid Constants
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_IDS   = {80, 85, 90, 95}
CANVAS_SIZE    = 900          # 900x900 px warped arena canvas
GRID_DIVISIONS = 12           # 12x12 cells -> 11x11 interior intersections
NUM_LINES      = GRID_DIVISIONS - 1
COLS           = list("ABCDEFGHIJK")
ROWS           = list(range(1, 12))

MARKER_INNER_CORNER = {
    80: 2,   # BR
    85: 3,   # BL
    90: 0,   # TL
    95: 1,   # TR
}

DEST_CORNERS = {
    80: np.array([0,             0],            dtype=np.float32),  # TL
    85: np.array([CANVAS_SIZE,   0],            dtype=np.float32),  # TR
    90: np.array([CANVAS_SIZE,   CANVAS_SIZE],  dtype=np.float32),  # BR
    95: np.array([0,             CANVAS_SIZE],  dtype=np.float32),  # BL
}


def detect_aruco(image):
    """Detect ArUco markers 80, 85, 90, 95."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=params)

    if ids is None:
        return {}

    detected = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        detected[int(marker_id)] = marker_corners.reshape(4, 2)
    return detected


def perspective_transform(image, marker_map):
    """Warp arena to top-down 900x900 canvas."""
    src_pts = []
    dst_pts = []
    for mid in sorted(REQUIRED_IDS):
        corners = marker_map[mid]
        inner_idx = MARKER_INNER_CORNER[mid]
        src_pts.append(corners[inner_idx])
        dst_pts.append(DEST_CORNERS[mid])

    M = cv2.getPerspectiveTransform(np.array(src_pts, dtype=np.float32),
                                    np.array(dst_pts, dtype=np.float32))
    warped = cv2.warpPerspective(image, M, (CANVAS_SIZE, CANVAS_SIZE),
                                 flags=cv2.INTER_LINEAR)
    return warped


def build_grid():
    """Build pixel coordinates and names for 11x11 intersections."""
    cell = CANVAS_SIZE / GRID_DIVISIONS  # 75.0 px
    xs = np.array([(c + 1) * cell for c in range(NUM_LINES)], dtype=np.float32)
    ys = np.array([(r + 1) * cell for r in range(NUM_LINES)], dtype=np.float32)

    grid_pts = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float32)
    names = []
    for r in range(NUM_LINES):
        row_names = []
        for c in range(NUM_LINES):
            grid_pts[r, c] = (xs[c], ys[r])
            row_names.append(f"{COLS[c]}{ROWS[r]}")
        names.append(row_names)

    return grid_pts, names, cell


def detect_all_entities(warped, grid_pts, names):
    """
    Detects:
    1. Critical survivors (3 red triangles)
    2. Stable survivors (3 yellow circles)
    3. Obstacles (3 black rectangles)
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    def nearest_intersection(cx, cy):
        best_name = None
        best_dist = float("inf")
        for r in range(NUM_LINES):
            for c in range(NUM_LINES):
                gx, gy = grid_pts[r, c]
                dist_sq = (cx - gx) ** 2 + (cy - gy) ** 2
                if dist_sq < best_dist:
                    best_dist = dist_sq
                    best_name = names[r][c]
        return best_name

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # --- 1. Critical Survivors (Red) ---
    red_lo1 = np.array([0,   120,  80], dtype=np.uint8)
    red_hi1 = np.array([8,   255, 255], dtype=np.uint8)
    red_lo2 = np.array([172, 120,  80], dtype=np.uint8)
    red_hi2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lo1, red_hi1),
        cv2.inRange(hsv, red_lo2, red_hi2)
    )
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crit_list = []
    for c in cnts_red:
        area = cv2.contourArea(c)
        if area >= 300.0:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                if 45.0 <= cx <= 855.0 and 45.0 <= cy <= 855.0:
                    crit_list.append({
                        'name': nearest_intersection(cx, cy),
                        'centroid': (cx, cy),
                        'area': area,
                        'contour': c
                    })
    crit_list.sort(key=lambda x: x['area'], reverse=True)
    crit_survivors = []
    seen_crit = set()
    for s in crit_list:
        if s['name'] not in seen_crit:
            seen_crit.add(s['name'])
            crit_survivors.append(s)
            if len(crit_survivors) == 3:
                break

    # --- 2. Stable Survivors (Yellow) ---
    yel_lo = np.array([20, 100,  80], dtype=np.uint8)
    yel_hi = np.array([38, 255, 255], dtype=np.uint8)
    mask_yel = cv2.inRange(hsv, yel_lo, yel_hi)
    mask_yel = cv2.morphologyEx(mask_yel, cv2.MORPH_OPEN, kernel)
    mask_yel = cv2.morphologyEx(mask_yel, cv2.MORPH_CLOSE, kernel)

    cnts_yel, _ = cv2.findContours(mask_yel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stab_list = []
    for c in cnts_yel:
        area = cv2.contourArea(c)
        if area >= 300.0:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                if 45.0 <= cx <= 855.0 and 45.0 <= cy <= 855.0:
                    stab_list.append({
                        'name': nearest_intersection(cx, cy),
                        'centroid': (cx, cy),
                        'area': area,
                        'contour': c
                    })
    stab_list.sort(key=lambda x: x['area'], reverse=True)
    stab_survivors = []
    seen_stab = set()
    for s in stab_list:
        if s['name'] not in seen_stab and s['name'] not in seen_crit:
            seen_stab.add(s['name'])
            stab_survivors.append(s)
            if len(stab_survivors) == 3:
                break

    # --- 3. Obstacles (Black) ---
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    margin_mask = np.zeros_like(thresh)
    margin_mask[60:840, 60:840] = 255
    dark_blobs = cv2.bitwise_and(thresh, margin_mask)

    cnts_dark, _ = cv2.findContours(dark_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    CLEARANCE_PX = 35.0  # Inflate obstacle by safety clearance
    for c in cnts_dark:
        area = cv2.contourArea(c)
        if area > 4500.0:
            x, y, w, h = cv2.boundingRect(c)
            box = (x - CLEARANCE_PX, y - CLEARANCE_PX, x + w + CLEARANCE_PX, y + h + CLEARANCE_PX)
            obstacles.append({
                'contour': c,
                'raw_bbox': (x, y, w, h),
                'inflated_box': box,
                'area': area
            })

    return crit_survivors, stab_survivors, obstacles


class PathPlanner:
    """A* Graph Planner with obstacle avoidance and shortcut smoothing."""

    def __init__(self, cell_size, obstacle_boxes):
        self.cell = cell_size
        self.obstacle_boxes = obstacle_boxes

    def is_point_in_obstacle(self, px, py):
        for bx1, by1, bx2, by2 in self.obstacle_boxes:
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                return True
        return False

    def line_intersects_box(self, p1, p2, box):
        x1, y1 = p1
        x2, y2 = p2
        bx1, by1, bx2, by2 = box
        if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
            return True
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / 4.0), 2)
        for i in range(steps + 1):
            t = i / steps
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                return True
        return False

    def is_line_clear(self, p1, p2):
        for box in self.obstacle_boxes:
            if self.line_intersects_box(p1, p2, box):
                return False
        return True

    def find_leg(self, start_pt, goal_pt):
        # If start_pt is inside an inflated obstacle margin (e.g. during a replan), snap to nearest free grid node
        if self.is_point_in_obstacle(start_pt[0], start_pt[1]):
            nodes_all = []
            for c in range(NUM_LINES):
                for r in range(NUM_LINES):
                    pt = ((c + 1) * self.cell, (r + 1) * self.cell)
                    if not self.is_point_in_obstacle(pt[0], pt[1]):
                        nodes_all.append(pt)
            if nodes_all:
                snap_pt = min(nodes_all, key=lambda p: math.hypot(p[0] - start_pt[0], p[1] - start_pt[1]))
                leg_from_snap = self.find_leg(snap_pt, goal_pt)
                return [start_pt] + leg_from_snap

        if self.is_line_clear(start_pt, goal_pt):
            return [start_pt, goal_pt]

        nodes = []
        for c in range(NUM_LINES):
            for r in range(NUM_LINES):
                pt = ((c + 1) * self.cell, (r + 1) * self.cell)
                if not self.is_point_in_obstacle(pt[0], pt[1]):
                    nodes.append((c, r, pt))

        pq = []
        heapq.heappush(pq, (math.hypot(goal_pt[0] - start_pt[0], goal_pt[1] - start_pt[1]), 0.0, start_pt, [start_pt]))
        visited = {}
        best_full_path = None

        while pq:
            f, g, curr, path = heapq.heappop(pq)
            if self.is_line_clear(curr, goal_pt):
                best_full_path = path + [goal_pt]
                break

            for c, r, npt in nodes:
                d_edge = math.hypot(npt[0] - curr[0], npt[1] - curr[1])
                max_reach = self.cell * 4.0 if curr == start_pt else self.cell * 2.8
                if d_edge <= max_reach and self.is_line_clear(curr, npt):
                    # Strongly discourage outer borders (Row 1, Row 11, Col A, Col K) to avoid flying over white borders
                    pen = 250.0 if (r in (0, 10) or c in (0, 10)) else 0.0
                    new_g = g + d_edge + pen
                    if (c, r) not in visited or visited[(c, r)] > new_g:
                        visited[(c, r)] = new_g
                        h = math.hypot(goal_pt[0] - npt[0], goal_pt[1] - npt[1])
                        heapq.heappush(pq, (new_g + h, new_g, npt, path + [npt]))

        if best_full_path is None:
            return [start_pt, goal_pt]

        # Greedy line-of-sight shortcutting
        smoothed = [best_full_path[0]]
        curr_idx = 0
        n = len(best_full_path)
        while curr_idx < n - 1:
            furthest_idx = curr_idx + 1
            for next_idx in range(n - 1, curr_idx, -1):
                if self.is_line_clear(best_full_path[curr_idx], best_full_path[next_idx]):
                    furthest_idx = next_idx
                    break
            smoothed.append(best_full_path[furthest_idx])
            curr_idx = furthest_idx
        return smoothed


class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')

        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()

        self.critical_waypoints = []
        self.stable_survivor_names = []
        self.obstacle_boxes = []
        self.warped_arena = None
        self.planner = None
        self.cell_size = 75.0

        self.route_ready = False
        self.frame_processed = False

        # Load grid coordinate mappings
        self.coords_map = self.load_coordinates()

        # Service 1: GetWaypoints on 'waypoints' (Critical route)
        self.srv_critical = self.create_service(
            GetWaypoints,
            'waypoints',
            self.critical_waypoints_callback,
            callback_group=self.cb_group
        )

        # Service 2: ReplanRoute on 'replan_route' (Stable route from drone position)
        self.srv_replan = self.create_service(
            ReplanRoute,
            'replan_route',
            self.replan_callback,
            callback_group=self.cb_group
        )

        # QoS profile matching camera publisher
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos,
            callback_group=self.cb_group
        )

        self.get_logger().info("Task 3A Waypoint Service initialized. Waiting for /image_raw...")

    def load_coordinates(self):
        """Load intersection coordinates mapping from JSON."""
        json_path = None
        try:
            pkg_share = get_package_share_directory('swift_pico')
            cand = os.path.join(pkg_share, 'scripts', 'grid_whycode_coordinates.json')
            if os.path.exists(cand):
                json_path = cand
        except Exception:
            pass

        if not json_path:
            local_cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_whycode_coordinates.json')
            if os.path.exists(local_cand):
                json_path = local_cand

        if not json_path or not os.path.exists(json_path):
            self.get_logger().error("Cannot find grid_whycode_coordinates.json!")
            return {}

        with open(json_path, 'r') as f:
            data = json.load(f)
        self.get_logger().info(f"Loaded {len(data)} grid coordinates from {json_path}")
        return data

    def name_to_px(self, name):
        c = COLS.index(name[0])
        r = int(name[1:]) - 1
        return ((c + 1) * self.cell_size, (r + 1) * self.cell_size)

    def whycode_to_px(self, wx, wy):
        px = 450.0 - (wx / 6.11) * 375.0
        py = 450.0 - (wy / 6.11) * 375.0
        return (px, py)

    def px_to_whycode(self, px, py):
        wx = (450.0 - px) / 375.0 * 6.11
        wy = (450.0 - py) / 375.0 * 6.11
        return (wx, wy)

    def image_callback(self, msg):
        if self.frame_processed:
            return

        self.get_logger().info(f"Received arena frame: {msg.width}x{msg.height}")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge conversion error: {e}")
            return

        # 1. Detect ArUco
        marker_map = detect_aruco(cv_image)
        missing = REQUIRED_IDS - set(marker_map.keys())
        if missing:
            self.get_logger().error(f"Missing corner markers: {missing}")
            return

        # 2. Perspective transform
        self.warped_arena = perspective_transform(cv_image, marker_map)

        # 3. Grid
        grid_pts, names, self.cell_size = build_grid()

        # 4. Detect Survivors and Obstacles
        crit_survivors, stab_survivors, obstacles = detect_all_entities(self.warped_arena, grid_pts, names)
        crit_names = [s['name'] for s in crit_survivors]
        self.stable_survivor_names = [s['name'] for s in stab_survivors]
        self.obstacle_boxes = [obs['inflated_box'] for obs in obstacles]

        self.get_logger().info(f"Detected {len(crit_survivors)} Critical Survivors: {crit_names}")
        self.get_logger().info(f"Detected {len(stab_survivors)} Stable Survivors: {self.stable_survivor_names}")
        self.get_logger().info(f"Detected {len(obstacles)} Black Obstacles.")

        # 5. Initialize Planner
        self.planner = PathPlanner(self.cell_size, self.obstacle_boxes)

        # 6. Plan Critical Route (Start at F6 -> 3 Critical Survivors)
        f6_px = self.name_to_px('F6')
        best_tour = None
        min_dist = float('inf')
        for perm in itertools.permutations(crit_names):
            curr = f6_px
            d = 0.0
            for name in perm:
                t_px = self.name_to_px(name)
                leg = self.planner.find_leg(curr, t_px)
                d += sum(math.hypot(leg[k+1][0] - leg[k][0], leg[k+1][1] - leg[k][1]) for k in range(len(leg)-1))
                curr = t_px
            if d < min_dist:
                min_dist = d
                best_tour = perm

        self.get_logger().info(f"Optimal Critical Route: {' -> '.join(best_tour)}")

        # Build detailed waypoints with obstacle bypass
        self.critical_waypoints = []
        curr = f6_px
        for name in best_tour:
            t_px = self.name_to_px(name)
            leg = self.planner.find_leg(curr, t_px)
            # Intermediate bypass points
            for pt in leg[1:-1]:
                wx, wy = self.px_to_whycode(pt[0], pt[1])
                self.critical_waypoints.append({
                    'x': float(wx), 'y': float(wy), 'z': 13.00,
                    'type': 1.0,  # 1.0 = Pass-through
                    'name': 'bypass'
                })
            # Survivor point
            coord = self.coords_map[name]
            self.critical_waypoints.append({
                'x': float(coord['x']), 'y': float(coord['y']), 'z': 13.00,
                'type': 0.0,  # 0.0 = Survivor (Visit + Save)
                'name': name
            })
            curr = t_px

        self.get_logger().info(f"Generated {len(self.critical_waypoints)} critical waypoints.")

        # Save initial visualization
        try:
            self.save_visualization(crit_survivors, stab_survivors, obstacles, self.critical_waypoints)
        except Exception as e:
            self.get_logger().warn(f"Failed to save route visualization: {e}")

        self.route_ready = True
        self.frame_processed = True
        self.destroy_subscription(self.image_sub)
        self.get_logger().info("Dropped /image_raw subscription. Ready to serve routes.")

    def critical_waypoints_callback(self, request, response):
        if not request.get_waypoints:
            return response

        wait_start = time.time()
        while not self.route_ready and (time.time() - wait_start < 25.0):
            time.sleep(0.05)

        if not self.route_ready:
            self.get_logger().error("Waypoints requested before route was ready!")
            return response

        response.waypoints.poses = []
        for wp in self.critical_waypoints:
            p = Pose()
            p.position.x = float(wp['x'])
            p.position.y = float(wp['y'])
            p.position.z = float(wp['z'])
            p.orientation.z = float(wp['type'])  # Encode type in orientation.z
            response.waypoints.poses.append(p)

        self.get_logger().info(f"Serving {len(response.waypoints.poses)} Critical Waypoints to client.")
        return response

    def replan_callback(self, request, response):
        """Plans the stable route from the drone's position at the Switch."""
        drone_pos = request.current_position.position
        start_wx, start_wy = drone_pos.x, drone_pos.y
        start_px = self.whycode_to_px(start_wx, start_wy)

        self.get_logger().info(f"Replan requested at Switch from position WhyCode=({start_wx:.2f}, {start_wy:.2f})")

        f6_px = self.name_to_px('F6')

        # Optimal TSP tour through stable survivors ending at F6
        best_tour = None
        min_dist = float('inf')
        for perm in itertools.permutations(self.stable_survivor_names):
            curr = start_px
            d = 0.0
            for name in perm:
                t_px = self.name_to_px(name)
                leg = self.planner.find_leg(curr, t_px)
                d += sum(math.hypot(leg[k+1][0] - leg[k][0], leg[k+1][1] - leg[k][1]) for k in range(len(leg)-1))
                curr = t_px
            leg_home = self.planner.find_leg(curr, f6_px)
            d += sum(math.hypot(leg_home[k+1][0] - leg_home[k][0], leg_home[k+1][1] - leg_home[k][1]) for k in range(len(leg_home)-1))
            if d < min_dist:
                min_dist = d
                best_tour = perm

        self.get_logger().info(f"Optimal Stable Route: {' -> '.join(best_tour)} -> F6")

        stable_waypoints = []
        curr = start_px
        for name in best_tour:
            t_px = self.name_to_px(name)
            leg = self.planner.find_leg(curr, t_px)
            for pt in leg[1:-1]:
                wx, wy = self.px_to_whycode(pt[0], pt[1])
                stable_waypoints.append({
                    'x': float(wx), 'y': float(wy), 'z': 13.00,
                    'type': 1.0,
                    'name': 'bypass'
                })
            coord = self.coords_map[name]
            stable_waypoints.append({
                'x': float(coord['x']), 'y': float(coord['y']), 'z': 13.00,
                'type': 0.0,
                'name': name
            })
            curr = t_px

        # Final leg to F6 for landing
        leg_f6 = self.planner.find_leg(curr, f6_px)
        for pt in leg_f6[1:-1]:
            wx, wy = self.px_to_whycode(pt[0], pt[1])
            stable_waypoints.append({
                'x': float(wx), 'y': float(wy), 'z': 13.00,
                'type': 1.0,
                'name': 'bypass'
            })
        coord_f6 = self.coords_map.get('F6', {'x': 0.0068, 'y': 0.0068})
        stable_waypoints.append({
            'x': float(coord_f6['x']), 'y': float(coord_f6['y']), 'z': 13.00,
            'type': 2.0,  # 2.0 = F6 Landing (Hover 1s + Land)
            'name': 'F6_LAND'
        })

        response.waypoints.poses = []
        for wp in stable_waypoints:
            p = Pose()
            p.position.x = float(wp['x'])
            p.position.y = float(wp['y'])
            p.position.z = float(wp['z'])
            p.orientation.z = float(wp['type'])
            response.waypoints.poses.append(p)

        self.get_logger().info(f"Serving {len(response.waypoints.poses)} Stable Waypoints ending at F6.")
        return response

    def save_visualization(self, crit_survivors, stab_survivors, obstacles, critical_wps):
        vis = self.warped_arena.copy()

        # Grid lines
        for i in range(1, GRID_DIVISIONS):
            pos = int(round(i * self.cell_size))
            cv2.line(vis, (pos, 0), (pos, CANVAS_SIZE), (40, 100, 40), 1)
            cv2.line(vis, (0, pos), (CANVAS_SIZE, pos), (40, 100, 40), 1)

        # Obstacles and clearance boxes
        for obs in obstacles:
            bx1, by1, bx2, by2 = [int(v) for v in obs['inflated_box']]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.drawContours(vis, [obs['contour']], -1, (20, 20, 20), -1)

        # Start F6
        f6_pt = (int(round(6 * self.cell_size)), int(round(6 * self.cell_size)))
        cv2.circle(vis, f6_pt, 8, (255, 255, 255), -1)
        cv2.putText(vis, "F6 (Start/Land)", (f6_pt[0] + 10, f6_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Survivors
        for s in crit_survivors:
            pt = (int(s['centroid'][0]), int(s['centroid'][1]))
            cv2.drawContours(vis, [s['contour']], -1, (0, 0, 255), 2)
            cv2.putText(vis, f"CRIT {s['name']}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        for s in stab_survivors:
            pt = (int(s['centroid'][0]), int(s['centroid'][1]))
            cv2.drawContours(vis, [s['contour']], -1, (0, 255, 255), 2)
            cv2.putText(vis, f"STAB {s['name']}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # Draw critical path
        curr_pt = f6_pt
        for wp in critical_wps:
            t_pt = self.whycode_to_px(wp['x'], wp['y'])
            t_pt = (int(round(t_pt[0])), int(round(t_pt[1])))
            cv2.line(vis, curr_pt, t_pt, (0, 140, 255), 2)
            if wp['type'] == 1.0:
                cv2.circle(vis, t_pt, 4, (0, 140, 255), -1)
            curr_pt = t_pt

        for target_dir in [os.path.join(WS_DIR, "task3a_data"), os.path.join(WS_DIR, "task_2c_data")]:
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, "planned_route.png")
            cv2.imwrite(out_path, vis)
            self.get_logger().info(f"Planned route visualization saved to {out_path}")


def main(args=None):
    rclpy.init(args=args)
    waypoints_node = WayPoints()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoints_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoints_node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
