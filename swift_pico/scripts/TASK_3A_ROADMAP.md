# eYRC Task 3A: Token-Optimized Master Execution Roadmap

> [!IMPORTANT]
> **Purpose of this Document:**
> This roadmap is specifically structured so an LLM agent (e.g., Claude Opus) can execute and maintain **Task 3A** with zero wasted tokens. Do **NOT** scan or explore the workspace. Everything needed is contained in the **3 core source files** and the **flight harness**.

---

## 1. Mission Brief & Official Scoring Rules (80 Marks)

### Objective
Fly the Swift Pico drone in simulation from **F6**, visit and save **3 Critical Survivors** before $t = 55.0\text{ s}$, trigger the **Switch**, visit and save **3 Stable Survivors** avoiding **3 Black Obstacles**, return to **F6**, hover, and land within the **150-second run cap**.

### Scoring Matrix (80 Marks Max)
| Element | Marks | Official Requirement | Evaluator Rule |
| :--- | :--- | :--- | :--- |
| **Visit** (6 survivors) | $3 \times 6 = \mathbf{18}$ | Hold at $Z = 13.0\text{ m}$ for $\ge 1.0\text{ s}$ | Continuous Euclidean dist $\le 0.40\text{ m}$ |
| **Save** (6 survivors) | $5 \times 6 = \mathbf{30}$ | Hold at $Z = 14.0\text{ m}$ for $\ge 2.0\text{ s}$ | Right after Visit, continuous dist $\le 0.40\text{ m}$ |
| **Hover at F6** | $\mathbf{3}$ | Hold at $Z = 13.0\text{ m}$ for $\ge 1.0\text{ s}$ | After last stable save, continuous dist $\le 0.40\text{ m}$ |
| **Landing at F6** | $\mathbf{3}$ | Land within $\pm 0.4\text{ m}$ on X, Y; $Z \ge 14.5\text{ m}$ | Must touch down before $150\text{ s}$; requires $\ge 1$ Visit |
| **Base Marks Subtotal** | $\mathbf{54 / 54}$ | **100% Guaranteed Base Marks** | Must achieve all 54 before chasing speed |
| **Speed Marks** | $\mathbf{26}$ | $\text{Speed} = 26 \times \frac{150 - T}{150} \times \frac{S}{6}$ | $T = \text{Landing Time (s)}$, $S = \text{Saves (0 to 6)}$ |
| **TOTAL SCORE** | $\mathbf{80}$ | $\text{Base} + \text{Speed}$ | **Competitive Target: ~64–69 / 80** |

---

## 2. The 3 Deadly Traps (Read Before Writing Any Code)

> [!CAUTION]
> **Trap 1: Euclidean Sphere vs. Cubic Box (The Major Flaw)**
> - **The Evaluator Rule:** $\sqrt{(x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2} \le 0.40\text{ m}$.
> - **The Trap:** Checking `abs(dx) <= 0.39 and abs(dy) <= 0.39 and abs(dz) <= 0.39`.
> - **Why It Fails:** At $(0.28, 0.22, 0.34)$, the box test passes, but Euclidean distance is $0.49\text{ m} > 0.40\text{ m}$! The server counts down, but the evaluator portal rejects it with `"visit hold too short: best 0.04 s of 1.00 s"`.
> - **The Fix:** In `pico_server.py`, `is_in_sphere` MUST compute exact Euclidean distance: `dist <= 0.35` (4–5 cm safety margin).

> [!WARNING]
> **Trap 2: Debounce Dropout Counter**
> - **The Evaluator Rule:** *"Leave the sphere or lose the marker and the hold starts again from zero"*.
> - **The Trap:** Having a dropout tolerance `if dropout_counter > 2: reset`.
> - **Why It Fails:** If the drone drifts out for even 1 frame, the evaluator resets its timer to 0, but the server timer keeps running. At J9, the server claims save complete, while the evaluator portal reports `"save hold too short: best 1.60 s of 2.00 s (3/8)"`.
> - **The Fix:** In `pico_server.py`, `hold_start = None` immediately if `not is_in_sphere`.

> [!WARNING]
> **Trap 3: The 55s Switch Cascade**
> - **The Evaluator Rule:** A stable survivor counts ONLY if visited *after* the Switch. The Switch occurs when all 3 critical saves are complete, or at $t = 55.00\text{ s}$.
> - **Why It Fails:** If the server fails even one critical hold (e.g., B10 or J9), the evaluator portal keeps the critical window open until $55.00\text{ s}$. When the drone reaches stable survivor B2 at $t = 51\text{ s}$, the portal rejects it: `"stable B2: visited before the switch only (doesn't count)"`.
> - **The Fix:** Ensure all 3 critical survivors achieve full $\ge 1.05\text{ s}$ Visit and $\ge 2.05\text{ s}$ Save inside $\le 0.35\text{ m}$ Euclidean sphere, finishing by $t \approx 38 - 40\text{ s}$.

---

## 3. Dedicated Project Files (Only Touch These)

Do not inspect or edit any other package. Only these 3 scripts and 2 harness scripts matter:

```
/home/eyepatch/eYRC/eYRC_26-27_Khoj-o-Drone/
├── src/swift_pico/src/waypoint_service.py   <-- Service (Vision, Obstacle Inflation, A* Routing)
├── src/swift_pico/src/pico_client.py        <-- Client (Liftoff detection, Switch timer, Action caller)
├── src/swift_pico/src/pico_server.py        <-- Server (PID controller, Euclidean sphere holds, Landing)
├── run_task_3a_experiment.py               <-- Automated 1-command test runner & zip packager
└── evaluate_task_3a.py                     <-- Local evaluator (mirrors official portal grading)
```

---

## 4. Implementation Specifications

### Script 1: `pico_server.py`
Located at `src/swift_pico/src/pico_server.py` and synced to `KD_2346_pico_server.py`.

1. **PID Controller Tuning (Proven Stable, No Oscillations):**
   ```python
   self.Kp = [68.0, 68.0, 62.0]
   self.Ki = [0.002, 0.002, 0.025]
   self.Kd = [98.0, 98.0, 48.0]
   self.cmd.rc_yaw = 1500  # Locked yaw rate (zero rotation)
   ```
2. **Euclidean Sphere Check:**
   ```python
   def is_in_sphere(self, target_xyz, radius=0.35):
       dist = math.sqrt(
           (self.curr_state[0] - target_xyz[0])**2 +
           (self.curr_state[1] - target_xyz[1])**2 +
           (self.curr_state[2] - target_xyz[2])**2
       )
       return dist <= radius
   ```
3. **Hold Logic (Strict Zero Debounce):**
   ```python
   def hold_at_target(self, target_xyz, hold_duration, goal_handle, feedback_msg):
       self.desired_state[0] = target_xyz[0]
       self.desired_state[1] = target_xyz[1]
       self.desired_state[2] = target_xyz[2]
       hold_start = None

       while True:
           if goal_handle.is_cancel_requested:
               return False

           now = time.time()
           # Publish feedback
           ...
           if self.is_in_sphere(target_xyz, radius=0.35):
               if hold_start is None:
                   hold_start = now
               if (now - hold_start) >= hold_duration:
                   return True
           else:
               hold_start = None  # Strict reset on exit
           time.sleep(0.02)
   ```
4. **Execution Durations:**
   - **Visit:** `1.05s` at $Z = 13.00$
   - **Save:** Set `desired_state[2] = 14.00`, then hold for `2.05s` at $Z = 14.00$
   - **Hover F6:** `1.05s` at $Z = 13.00$
   - **Land F6:** Direct descent to $Z = 14.85$, break when `curr_state[2] >= 14.75`, set throttle to 1000 and disarm.

---

### Script 2: `pico_client.py`
Located at `src/swift_pico/src/pico_client.py` and synced to `KD_2346_pico_client.py`.

1. **Liftoff Detection:**
   - Listen to `/whycode_node/markers`. Start mission timer ($t = 0$) at first frame where $z < 14.60\text{ m}$.
2. **Critical Route Execution:**
   - Sequential dispatch of goals received from `GetWaypoints`.
3. **The Switch Trigger:**
   - Trigger condition: `self.critical_saves_count >= 3 or (time.time() - self.liftoff_time) >= 54.5`.
   - If a goal is active: call `cancel_goal_async()`.
   - Once cancelled or idle: call `replan_route` with `current_position = self.latest_pos`.
4. **Stable Route Execution:**
   - Dispatch stable waypoints received from `replan_route`, concluding with F6 landing goal.

---

### Script 3: `waypoint_service.py`
Located at `src/swift_pico/src/waypoint_service.py` and synced to `KD_2346_waypoint_service.py`.

1. **Perception Pipeline:**
   - Detect ArUco corners (80, 85, 90, 95) $\to$ Warp to $900 \times 900\text{ px}$.
   - Detect 3 Critical Survivors (Red triangles: hue 0–10 & 170–180).
   - Detect 3 Stable Survivors (Yellow circles: hue 20–35).
   - Detect 3 Black Obstacles (Threshold $< 40$, area $> 4500\text{ px}^2$).
2. **Boundary & Obstacle Safety:**
   - Inflate each obstacle bounding box by `CLEARANCE_PX = 35.0`.
   - In `is_line_clear`, reject any line with points $px < 65$ or $px > 835$ or $py < 65$ or $py > 835$ (prevents touching outer white borders).
3. **Route Planning:**
   - Start at F6 $\to$ TSP permutation for shortest critical path $\to$ A* graph search avoiding inflated obstacles $\to$ Line-of-sight shortcutting.
   - On `replan_route`, plan from drone's current $(x, y)$ through all 3 stable survivors, ending at F6.

---

## 5. Execution & Verification Workflow

### Step 1: Build the Workspace
```bash
cd /home/eyepatch/eYRC/eYRC_26-27_Khoj-o-Drone
colcon build --packages-select swift_pico waypoint_navigation
source install/setup.bash
```

### Step 2: Run End-to-End Simulation (1 Command)
Run the automated runner which launches simulation, service, client, server, records rosbag, evaluates score, and packages the zip:
```bash
python3 run_task_3a_experiment.py
```

### Step 3: Run Standalone Evaluation
To evaluate any recorded bag file against the exact portal rubric:
```bash
python3 evaluate_task_3a.py task_3a
```

### Step 4: Verify Submission ZIP Structure
The required submission file is `KD_2346_task_3a.zip`.
Verify that it contains **exactly 5 files at the zip root**:
```bash
zipinfo KD_2346_task_3a.zip
```
Expected output:
```
Archive:  KD_2346_task_3a.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
    ...    ...        ...     KD_2346_waypoint_service.py
    ...    ...        ...     KD_2346_pico_client.py
    ...    ...        ...     KD_2346_pico_server.py
  4386816  ...        ...     task_3a_0.db3
     1983  ...        ...     metadata.yaml
---------                     -------
                              5 files
```

---

## 6. Token-Saving Rule Summary for Claude Opus

1. **Do NOT search the workspace:** The only files to read or edit are:
   - `src/swift_pico/src/pico_server.py`
   - `src/swift_pico/src/pico_client.py`
   - `src/swift_pico/src/waypoint_service.py`
2. **Do NOT inspect simulator internals:** MuJoCo bridge, WhyCode nodes, and rotor plugins are pre-configured and must remain untouched.
3. **Do NOT modify PID gains:** $K_p = [68, 68, 62]$, $K_d = [98, 98, 48]$ are already optimal.
4. **Always copy scripts to submission names:**
   ```bash
   cp src/swift_pico/src/waypoint_service.py KD_2346_waypoint_service.py
   cp src/swift_pico/src/pico_client.py KD_2346_pico_client.py
   cp src/swift_pico/src/pico_server.py KD_2346_pico_server.py
   ```
5. **Score Verification Target:**
   - **Base Marks:** 54.00 / 54.00
   - **Landing Time:** $T \le 92\text{ s}$
   - **Total Score:** $\ge 64.0\text{ / }80.0$
