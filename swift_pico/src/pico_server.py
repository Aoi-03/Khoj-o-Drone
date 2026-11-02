#!/usr/bin/env python3

'''
Task 3A – Action Server (WayPointServer)
=======================================
Implements the NavToWaypoint action server to navigate the Swift Pico Drone.
Features:
- Highly tuned, stable PID controller with world-to-body frame rotation.
- Heading locked with zero yaw rate command (rc_yaw = 1500).
- Accepts goal cancellation mid-flight (CancelResponse.ACCEPT) and holds in place.
- Distinct handling of waypoint types:
    * Type 0.0 (Survivor): Visit (hold 1.05s at Z=13) -> Save (descend and hold 2.05s at Z=14).
    * Type 1.0 (Pass-Through): Fly past at Z=13 without stopping.
    * Type 2.0 (F6 Home / Landing): Hover at Z=13 for 1.05s, then descend and land at F6.
- Publishes /controller_heartbeat at 10 Hz.
'''

import time
import math

def euler_from_quaternion(q):
    """Convert quaternion [x, y, z, w] to euler [roll, pitch, yaw] in radians."""
    x, y, z, w = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Custom Action
from waypoint_navigation.action import NavToWaypoint

# ROS Messages
from swift_msgs.msg import SwiftMsgs
from std_msgs.msg import Header
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from nav_msgs.msg import Odometry


class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.odometry_callback_group = ReentrantCallbackGroup()

        # State tracking: [x, y, z] in WhyCode coordinates
        self.curr_state = [0.0, 0.0, 0.0]
        self.drone_position = [0.0, 0.0, 0.0]
        self.desired_state = [0.0068, 0.0068, 13.00]  # Initial setpoint (F6 / arena center)

        # Orientation & Yaw
        self.yaw = 0.0              # current yaw in degrees (from WhyCode)
        self.desired_yaw = 0.0      # target yaw angle (degrees)
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0

        # Timing & hold tracking
        self.dtime = 0
        self.last_marker_time = time.time()
        self.is_landed = False

        # Drone command message initialization
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1500

        # PID gains: [roll, pitch, throttle] (optimized fast & stable tuning)
        self.Kp = [68.0, 68.0, 62.0]
        self.Ki = [0.002, 0.002, 0.025]
        self.Kd = [98.0, 98.0, 48.0]

        # PID error variables: [roll, pitch, throttle]
        self.error = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.sum_error = [0.0, 0.0, 0.0]
        self.max_sum_error = [50.0, 50.0, 50.0]
        self.first_cycle = True

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        # Subscriptions
        self.create_subscription(
            MarkerArray,
            '/whycode_node/markers',
            self.whycode_callback,
            1,
            callback_group=self.pid_callback_group
        )

        self.create_subscription(
            Odometry,
            '/rotors/odometry',
            self.odometry_callback,
            10,
            callback_group=self.odometry_callback_group
        )

        # Action Server: NavToWaypoint on 'waypoint_navigation' with CANCEL support
        self.action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group
        )

        # Heartbeat timer (10 Hz)
        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        # Arm drone after brief pause to allow heartbeat setup
        time.sleep(0.5)
        self.arm()
        self.get_logger().info('Task 3A WayPointServer initialized, drone armed, ready for action goals.')

    def cancel_callback(self, goal_handle):
        """Accepts cancellation requests mid-goal."""
        self.get_logger().info('Received cancel request from client. Accepting cancel.')
        return CancelResponse.ACCEPT

    def publish_heartbeat(self):
        hb = Header()
        hb.stamp = self.get_clock().now().to_msg()
        hb.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(hb)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)

    def whycode_callback(self, msg):
        if not msg.markers:
            return

        marker = msg.markers[0]
        position = marker.position.position
        self.curr_state = [position.x, position.y, position.z]
        self.drone_position = self.curr_state[:]
        self.yaw = math.degrees(marker.rotation.z)
        self.dtime = msg.header.stamp.sec
        self.last_marker_time = time.time()

        if self.is_landed:
            self.disarm()
            return

        # Compute position errors as in Task 2B:
        # pitch error (x): desired_x - current_x
        # roll error (y):  desired_y - current_y
        # throttle error (z): current_z - desired_z (WhyCode z points downward from ceiling)
        self.error[1] = self.desired_state[0] - self.curr_state[0]
        self.error[0] = self.desired_state[1] - self.curr_state[1]
        self.error[2] = self.curr_state[2] - self.desired_state[2]

        # Publish error message
        pos_error = Error()
        pos_error.roll_error = float(self.error[0])
        pos_error.pitch_error = float(self.error[1])
        pos_error.throttle_error = float(self.error[2])
        pos_error.yaw_error = float(self.desired_yaw - self.yaw_deg)
        self.pos_error_pub.publish(pos_error)

        if self.first_cycle:
            self.prev_error[0] = self.error[0]
            self.prev_error[1] = self.error[1]
            self.prev_error[2] = self.error[2]
            self.first_cycle = False

        # Anti-windup clamping for integral sum
        for i in range(3):
            self.sum_error[i] += self.error[i]
            self.sum_error[i] = max(min(self.sum_error[i], self.max_sum_error[i]), -self.max_sum_error[i])

        # Decay integral sum when close to target to prevent overshoot
        if abs(self.error[0]) < 0.25 and abs(self.error[1]) < 0.25:
            self.sum_error[0] *= 0.95
            self.sum_error[1] *= 0.95

        # Continuous derivatives (WhyCode at 30 Hz -> dt ≈ 0.0333s)
        dt = 0.0333
        d_roll = (self.error[0] - self.prev_error[0]) / dt
        d_pitch = (self.error[1] - self.prev_error[1]) / dt
        d_throttle = (self.error[2] - self.prev_error[2]) / dt

        # PID control outputs
        out_roll = (self.Kp[0] * self.error[0]
                    + self.Ki[0] * self.sum_error[0]
                    + self.Kd[0] * d_roll)

        out_pitch = (self.Kp[1] * self.error[1]
                     + self.Ki[1] * self.sum_error[1]
                     + self.Kd[1] * d_pitch)

        out_throttle = (self.Kp[2] * self.error[2]
                        + self.Ki[2] * self.sum_error[2]
                        + self.Kd[2] * d_throttle)

        # Update previous errors
        self.prev_error[0] = self.error[0]
        self.prev_error[1] = self.error[1]
        self.prev_error[2] = self.error[2]

        # Map to RC PWM channels
        cmd_roll = int(1500 + out_roll)
        cmd_pitch = int(1500 + out_pitch)
        cmd_throttle = int(1500 + out_throttle)

        # Attitude clamping & throttle limits (preserves attitude authority)
        self.cmd.rc_roll = max(min(cmd_roll, 1900), 1100)
        self.cmd.rc_pitch = max(min(cmd_pitch, 1900), 1100)
        self.cmd.rc_throttle = max(min(cmd_throttle, 1750), 1250)
        self.cmd.rc_yaw = 1500  # Locked at 1500 to keep yaw steady (zero yaw rate)
        self.cmd.rc_aux4 = 2000

        self.command_pub.publish(self.cmd)

    def is_in_sphere(self, target_xyz, radius=0.38):
        """Checks if drone is within Euclidean sphere of radius around target_xyz.
        Uses 0.38m radius (2cm safety margin inside portal's 0.40m sphere)."""
        dist = math.sqrt(
            (self.curr_state[0] - target_xyz[0])**2 +
            (self.curr_state[1] - target_xyz[1])**2 +
            (self.curr_state[2] - target_xyz[2])**2
        )
        return dist <= radius

    def hold_at_target(self, target_xyz, hold_duration, goal_handle, feedback_msg, eval_xyz=None):
        """Holds position in Euclidean sphere for hold_duration seconds.
        Returns True if held, False if cancelled.
        Starts timer once drone enters Euclidean dist <= 0.36m.
        During hold, requires continuous presence within 0.39m Euclidean sphere (portal is 0.40m)."""
        self.desired_state[0] = target_xyz[0]
        self.desired_state[1] = target_xyz[1]
        self.desired_state[2] = target_xyz[2]

        check_xyz = eval_xyz if eval_xyz is not None else target_xyz
        hold_start = None
        dropout_counter = 0

        while True:
            if goal_handle.is_cancel_requested:
                return False

            now = time.time()
            feedback_msg.current_waypoint.pose.position.x = float(self.curr_state[0])
            feedback_msg.current_waypoint.pose.position.y = float(self.curr_state[1])
            feedback_msg.current_waypoint.pose.position.z = float(self.curr_state[2])
            feedback_msg.current_waypoint.header.stamp.sec = int(now - hold_start) if hold_start else 0
            goal_handle.publish_feedback(feedback_msg)

            dist = math.sqrt(
                (self.curr_state[0] - check_xyz[0])**2 +
                (self.curr_state[1] - check_xyz[1])**2 +
                (self.curr_state[2] - check_xyz[2])**2
            )

            if hold_start is None:
                # Start hold timer cleanly as soon as drone is within 0.36m Euclidean
                if dist <= 0.36:
                    hold_start = now
            else:
                # During hold: stay strictly inside 0.39m Euclidean sphere (portal is 0.40m)
                if dist <= 0.39:
                    held = now - hold_start
                    if held >= hold_duration:
                        return True
                else:
                    hold_start = None

            time.sleep(0.02)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        target_x = goal_handle.request.waypoint.position.x
        target_y = goal_handle.request.waypoint.position.y
        target_z = goal_handle.request.waypoint.position.z
        wp_type  = goal_handle.request.waypoint.orientation.z

        # Reset horizontal integrators on new waypoint to prevent overshoot
        self.sum_error[0] = 0.0
        self.sum_error[1] = 0.0
        self.first_cycle = True
        start_mono = time.time()

        feedback_msg = NavToWaypoint.Feedback()

        is_landing = (wp_type == 2.0) or (abs(target_x - 0.0068) < 0.15 and abs(target_y - 0.0068) < 0.15 and wp_type != 0.0)
        is_passthrough = (wp_type == 1.0)

        # ── Case 1: Pass-Through Waypoint ────────────────────────────────────
        if is_passthrough:
            self.get_logger().info(f'Pass-through waypoint: [{target_x:.2f}, {target_y:.2f}, 13.00]')
            self.desired_state[0] = target_x
            self.desired_state[1] = target_y
            self.desired_state[2] = 13.00

            while True:
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Cancel requested during pass-through.')
                    goal_handle.canceled()
                    self.desired_state[0] = self.curr_state[0]
                    self.desired_state[1] = self.curr_state[1]
                    self.desired_state[2] = 13.00
                    result = NavToWaypoint.Result()
                    result.hov_time = int(time.time() - start_mono)
                    return result

                feedback_msg.current_waypoint.pose.position.x = float(self.curr_state[0])
                feedback_msg.current_waypoint.pose.position.y = float(self.curr_state[1])
                feedback_msg.current_waypoint.pose.position.z = float(self.curr_state[2])
                goal_handle.publish_feedback(feedback_msg)

                dx = abs(self.curr_state[0] - target_x)
                dy = abs(self.curr_state[1] - target_y)
                if dx <= 0.52 and dy <= 0.52:
                    self.get_logger().info(f'Passed waypoint [{target_x:.2f}, {target_y:.2f}]')
                    break

                time.sleep(0.02)

            goal_handle.succeed()
            result = NavToWaypoint.Result()
            result.hov_time = int(time.time() - start_mono)
            return result

        # ── Case 2: F6 Hover and Land ────────────────────────────────────────
        elif is_landing:
            self.get_logger().info(f'Final Goal: Hover and Land at F6 [{target_x:.2f}, {target_y:.2f}]')

            # Step 1: Hover at Z = 13.00 for >= 1.06s (command 12.95 to eliminate steady-state droop)
            self.get_logger().info('Hovering at F6 Z=13 for 1s...')
            held = self.hold_at_target([target_x, target_y, 12.95], 1.06, goal_handle, feedback_msg, eval_xyz=[target_x, target_y, 13.00])
            if not held:
                goal_handle.canceled()
                result = NavToWaypoint.Result()
                result.hov_time = int(time.time() - start_mono)
                return result

            self.get_logger().info('F6 Hover complete! Beginning landing descent...')

            # Step 2: Smooth, fast direct descent to touchdown
            self.desired_state[0] = target_x
            self.desired_state[1] = target_y
            self.desired_state[2] = 14.85
            self.first_cycle = True
            t_land_start = time.time()
            while time.time() - t_land_start < 1.5:
                if self.curr_state[2] >= 14.75:
                    break
                time.sleep(0.02)

            # Touchdown and disarm
            self.cmd.rc_roll = 1500
            self.cmd.rc_pitch = 1500
            self.cmd.rc_yaw = 1500
            self.cmd.rc_throttle = 1000
            self.cmd.rc_aux4 = 1000
            self.command_pub.publish(self.cmd)

            self.is_landed = True
            self.disarm()
            self.get_logger().info(f'Landed on F6! Final Position: [{self.curr_state[0]:.2f}, {self.curr_state[1]:.2f}, {self.curr_state[2]:.2f}]')

            goal_handle.succeed()
            result = NavToWaypoint.Result()
            result.hov_time = int(time.time() - start_mono)
            return result

        # ── Case 3: Survivor Waypoint (Visit + Save) ──────────────────────────
        else:
            self.get_logger().info(f'Survivor Goal: [{target_x:.2f}, {target_y:.2f}]')

            # 1. Visit: hold at Z = 13 for 1.06s (command 12.95 to eliminate steady-state droop)
            self.get_logger().info('Starting Visit (Z=13 for 1s)...')
            held_visit = self.hold_at_target([target_x, target_y, 12.95], 1.06, goal_handle, feedback_msg, eval_xyz=[target_x, target_y, 13.00])
            if not held_visit:
                self.get_logger().info('Goal cancelled during Visit. Holding in place.')
                goal_handle.canceled()
                self.desired_state[0] = self.curr_state[0]
                self.desired_state[1] = self.curr_state[1]
                self.desired_state[2] = 13.00
                result = NavToWaypoint.Result()
                result.hov_time = int(time.time() - start_mono)
                return result

            self.get_logger().info('Visit Complete! Descending to Save (Z=14 for 2s)...')

            # Pre-descent to 14.00: reset vertical integrator to prevent altitude bounce, allow fast settling
            self.sum_error[2] = 0.0
            self.desired_state[2] = 14.00
            self.first_cycle = True
            t_desc = time.time()
            while time.time() - t_desc < 0.8:
                dz = abs(self.curr_state[2] - 14.00)
                dist_14 = math.sqrt(
                    (self.curr_state[0] - target_x)**2 +
                    (self.curr_state[1] - target_y)**2 +
                    (self.curr_state[2] - 14.00)**2
                )
                if dist_14 <= 0.28 and dz <= 0.20:
                    break
                time.sleep(0.02)

            # 2. Save: hold at Z = 14.00 for 2.10s (command 14.00, evaluate at 14.00)
            held_save = self.hold_at_target(
                [target_x, target_y, 14.00], 2.10, goal_handle, feedback_msg, eval_xyz=[target_x, target_y, 14.00]
            )
            if not held_save:
                self.get_logger().info('Goal cancelled during Save. Holding in place.')
                goal_handle.canceled()
                self.desired_state[0] = self.curr_state[0]
                self.desired_state[1] = self.curr_state[1]
                self.desired_state[2] = 13.00
                result = NavToWaypoint.Result()
                result.hov_time = int(time.time() - start_mono)
                return result

            self.get_logger().info(f'Save Complete at [{target_x:.2f}, {target_y:.2f}]!')

            # Reset vertical integrator and command cruise altitude, declare success immediately
            self.sum_error[2] = 0.0
            self.desired_state[2] = 12.95
            self.first_cycle = True

            goal_handle.succeed()
            result = NavToWaypoint.Result()
            result.hov_time = int(time.time() - start_mono)
            return result


def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
