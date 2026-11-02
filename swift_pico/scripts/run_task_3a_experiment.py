#!/usr/bin/env python3
"""
End-to-end flight runner and packaging script for Task 3A.
Steps:
1. Start simulation: ros2 launch swift_pico swift_pico_simulation.launch.py
2. Run waypoint_service: ros2 run swift_pico waypoint_service (wait until ready)
3. Run pico_client: ros2 run swift_pico pico_client
4. Start rosbag: ros2 bag record -o task_3a /whycode_node/markers /controller_heartbeat /ros_check
5. Run pico_server: ros2 run swift_pico pico_server
6. Monitor flight until landing, evaluate rosbag, and package KD_2346_task_3a.zip.
"""

import os
import sys
import time
import shutil
import subprocess
import zipfile

WS_DIR = '/home/eyepatch/eYRC/eYRC_26-27_Khoj-o-Drone'
SETUP_BASH = os.path.join(WS_DIR, 'install/setup.bash')

def cleanup():
    print("[runner] Cleaning up lingering processes and shm...")
    subprocess.run(
        "killall -9 mujoco_bridge whycode_node image_view roll_pitch_yawrate_thrust_controller_node rotors_swift_interface 2>/dev/null || true",
        shell=True, check=False
    )
    subprocess.run("pkill -9 -f waypoint_service 2>/dev/null || true", shell=True, check=False)
    subprocess.run("pkill -9 -f pico_client 2>/dev/null || true", shell=True, check=False)
    subprocess.run("pkill -9 -f pico_server 2>/dev/null || true", shell=True, check=False)
    subprocess.run("pkill -9 -f KD_2346 2>/dev/null || true", shell=True, check=False)
    subprocess.run("pkill -9 -f ros2 2>/dev/null || true", shell=True, check=False)
    subprocess.run("rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/swift_pico_image_raw 2>/dev/null || true", shell=True, check=False)
    time.sleep(1.5)

def main():
    cleanup()

    # Capture environment
    env = os.environ.copy()
    cmd = f"source /opt/ros/humble/setup.bash && source {SETUP_BASH} && env"
    res = subprocess.run(f"bash -c '{cmd}'", shell=True, stdout=subprocess.PIPE, text=True)
    for line in res.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            env[k] = v

    fastdds_profile_path = os.path.join(WS_DIR, 'install/swift_pico/share/swift_pico/config/fastdds_profile.xml')
    if os.path.exists(fastdds_profile_path):
        env['FASTRTPS_DEFAULT_PROFILES_FILE'] = fastdds_profile_path

    os.chdir(WS_DIR)

    # Clean old bag
    if os.path.exists("task_3a"):
        shutil.rmtree("task_3a")

    # 1. Start simulation
    print("[runner] 1/6 Starting simulation...")
    sim_log = open("sim_3a.log", "w")
    sim_proc = subprocess.Popen(
        ["ros2", "launch", "swift_pico", "swift_pico_simulation.launch.py"],
        env=env,
        stdout=sim_log,
        stderr=subprocess.STDOUT
    )

    # 2. Wait for simulation markers
    print("[runner] Waiting for simulation to initialize and markers to appear...")
    start_wait = time.time()
    sim_ready = False
    while time.time() - start_wait < 35.0:
        check = subprocess.run(
            ["ros2", "topic", "echo", "/whycode_node/markers", "--once"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=6.0
        )
        if "markers:" in check.stdout and "position:" in check.stdout:
            print("[runner] Marker detected! Simulation is active.")
            sim_ready = True
            break
        time.sleep(1.0)

    if not sim_ready:
        print("[runner] ERROR: Simulation failed to publish markers in time!")
        sim_proc.kill()
        cleanup()
        sys.exit(1)

    time.sleep(2.0)

    # 3. Start Waypoint Service
    print("[runner] 2/6 Starting waypoint_service...")
    svc_log = open("service_3a.log", "w")
    svc_proc = subprocess.Popen(
        ["ros2", "run", "swift_pico", "waypoint_service"],
        env=env,
        stdout=svc_log,
        stderr=subprocess.STDOUT
    )

    # Wait for route to be planned
    print("[runner] Waiting for waypoint_service to process frame and plan route...")
    start_wp = time.time()
    while time.time() - start_wp < 25.0:
        if os.path.exists("service_3a.log"):
            with open("service_3a.log", "r") as f:
                content = f.read()
                if "Ready to serve routes" in content or "Optimal Critical Route" in content:
                    print("[runner] Route planned and ready!")
                    break
        time.sleep(0.5)

    # 4. Start Action Client
    print("[runner] 3/6 Starting pico_client...")
    client_log = open("client_3a.log", "w")
    client_proc = subprocess.Popen(
        ["ros2", "run", "swift_pico", "pico_client"],
        env=env,
        stdout=client_log,
        stderr=subprocess.STDOUT
    )
    time.sleep(2.0)

    # 5. Start rosbag recording (while drone is still at start position)
    print("[runner] 4/6 Starting rosbag recording...")
    bag_proc = subprocess.Popen(
        ["ros2", "bag", "record", "-o", "task_3a", "/whycode_node/markers", "/controller_heartbeat", "/ros_check"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    time.sleep(2.0)

    # 6. Start Action Server (kicks off navigation)
    print("[runner] 5/6 Starting pico_server (Navigation starts!)...")
    server_log = open("server_3a.log", "w")
    server_proc = subprocess.Popen(
        ["ros2", "run", "swift_pico", "pico_server"],
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT
    )

    # 7. Monitor flight execution
    print("[runner] 6/6 Monitoring flight progress...")
    flight_start = time.time()
    flight_completed = False

    while time.time() - flight_start < 160.0:
        elapsed = time.time() - flight_start

        # Check client log for completion
        if os.path.exists("client_3a.log"):
            with open("client_3a.log", "r") as f:
                c_content = f.read()
                if "MISSION COMPLETE" in c_content:
                    print(f"\n[runner] MISSION COMPLETE detected in client log at {elapsed:.1f}s!")
                    flight_completed = True
                    break

        # Check server log for landing
        if os.path.exists("server_3a.log"):
            with open("server_3a.log", "r") as f:
                s_content = f.read()
                if "Landed on F6" in s_content:
                    print(f"\n[runner] Drone landing detected in server log at {elapsed:.1f}s!")
                    flight_completed = True
                    break

        # Print heartbeat progress
        sys.stdout.write(f"\r[runner] Flight time: {elapsed:.1f}s / 150s max ...")
        sys.stdout.flush()
        time.sleep(1.0)

    print("\n[runner] Settling for 3 seconds post-landing...")
    time.sleep(3.0)

    # Stop rosbag cleanly
    print("[runner] Stopping rosbag recording...")
    bag_proc.send_signal(subprocess.signal.SIGINT)
    try:
        bag_proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        bag_proc.kill()

    # Terminate processes
    print("[runner] Terminating simulation and nodes...")
    server_proc.kill()
    client_proc.kill()
    svc_proc.kill()
    sim_proc.kill()
    cleanup()

    # 8. Evaluate rosbag
    print("\n[runner] Evaluating recorded rosbag...")
    eval_script = os.path.join(WS_DIR, "evaluate_task_3a.py")
    subprocess.run([sys.executable, eval_script, "task_3a"])

    # 9. Package Submission ZIP
    print("\n[runner] Packaging KD_2346_task_3a.zip...")
    zip_name = "KD_2346_task_3a.zip"
    if os.path.exists(zip_name):
        os.remove(zip_name)

    # Copy latest scripts to submission names
    shutil.copy("src/swift_pico/src/waypoint_service.py", "KD_2346_waypoint_service.py")
    shutil.copy("src/swift_pico/src/pico_client.py", "KD_2346_pico_client.py")
    shutil.copy("src/swift_pico/src/pico_server.py", "KD_2346_pico_server.py")

    bag_db3 = "task_3a/task_3a_0.db3"
    bag_yaml = "task_3a/metadata.yaml"

    if os.path.exists(bag_db3) and os.path.exists(bag_yaml):
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write("KD_2346_waypoint_service.py", arcname="KD_2346_waypoint_service.py")
            z.write("KD_2346_pico_client.py", arcname="KD_2346_pico_client.py")
            z.write("KD_2346_pico_server.py", arcname="KD_2346_pico_server.py")
            z.write(bag_db3, arcname="task_3a_0.db3")
            z.write(bag_yaml, arcname="metadata.yaml")

        print(f"[runner] Successfully created {zip_name}!")
        subprocess.run(["zipinfo", zip_name])
    else:
        print(f"[runner] ERROR: Rosbag files {bag_db3} or {bag_yaml} not found!")

if __name__ == '__main__':
    main()
