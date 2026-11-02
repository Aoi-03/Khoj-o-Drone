#!/usr/bin/env python3
"""
Task 3A – Action Client (WayPointClient)
=======================================
Orchestrates the entire Task 3A mission:
1. Calls GetWaypoints service to fetch the critical route.
2. Waits for action server 'waypoint_navigation'.
3. Starts the critical window clock at liftoff (t = 0 when z < 14.6).
4. Executes Critical Phase: visits and saves critical survivors.
5. Monitors Switch condition: all 3 critical saved OR elapsed time >= 54.5s.
6. The Switch:
   - Cancels active goal in flight if present.
   - Captures current drone position.
   - Calls replan_route service to get the stable route avoiding obstacles.
7. Executes Stable Phase: visits and saves 3 stable survivors.
8. Home: flies back to F6, hovers, and lands.
"""

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose
from whycode_interfaces.msg import MarkerArray
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints, ReplanRoute


class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')

        self.cb_group = ReentrantCallbackGroup()

        # Phase tracking: 'INIT', 'CRITICAL', 'SWITCHING', 'STABLE', 'COMPLETED'
        self.phase = 'INIT'

        self.critical_goals = []
        self.stable_goals = []
        self.current_goal_list = []
        self.goal_index = 0

        self.current_goal_handle = None
        self.is_goal_active = False

        self.liftoff_detected = False
        self.liftoff_time = None

        self.critical_saves_count = 0
        self.stable_saves_count = 0

        # Latest drone coordinates
        self.latest_pos = [0.0068, 0.0068, 13.00]

        # Action Client
        self.action_client = ActionClient(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            callback_group=self.cb_group
        )

        # Service Clients
        self.cli_get_waypoints = self.create_client(
            GetWaypoints,
            'waypoints',
            callback_group=self.cb_group
        )

        self.cli_replan = self.create_client(
            ReplanRoute,
            'replan_route',
            callback_group=self.cb_group
        )

        # Marker Subscription for fast liftoff & position updates
        self.create_subscription(
            MarkerArray,
            '/whycode_node/markers',
            self.marker_callback,
            1,
            callback_group=self.cb_group
        )

        # Critical timer monitor (runs at 10 Hz)
        self.monitor_timer = self.create_timer(
            0.1,
            self.mission_monitor_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info('Task 3A WayPointClient initialized.')

        # Kick off waypoints fetching asynchronously
        self.fetch_timer = self.create_timer(
            0.15,
            self.try_fetch_waypoints,
            callback_group=self.cb_group
        )

    def marker_callback(self, msg):
        if not msg.markers:
            return
        p = msg.markers[0].position.position
        self.latest_pos = [p.x, p.y, p.z]

        if not self.liftoff_detected and p.z < 14.6:
            self.liftoff_detected = True
            self.liftoff_time = time.time()
            self.get_logger().info('*** LIFTOFF DETECTED! Critical window timer started (t = 0) ***')

    def try_fetch_waypoints(self):
        if not self.cli_get_waypoints.service_is_ready():
            self.get_logger().info('Waiting for waypoints service...')
            return

        self.fetch_timer.cancel()

        self.get_logger().info('Requesting critical route from waypoints service...')
        req = GetWaypoints.Request()
        req.get_waypoints = True
        future = self.cli_get_waypoints.call_async(req)
        future.add_done_callback(self.waypoints_received_callback)

    def waypoints_received_callback(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'GetWaypoints call failed: {e}')
            return

        if not response.waypoints.poses:
            self.get_logger().warn('Received 0 waypoints from service. Retrying in 1s...')
            self.fetch_timer = self.create_timer(
                1.0,
                self.try_fetch_waypoints,
                callback_group=self.cb_group
            )
            return

        self.get_logger().info(f'Received {len(response.waypoints.poses)} critical route waypoints.')

        for p in response.waypoints.poses:
            self.critical_goals.append({
                'x': p.position.x,
                'y': p.position.y,
                'z': p.position.z,
                'type': p.orientation.z
            })

        self.current_goal_list = self.critical_goals
        self.goal_index = 0
        self.phase = 'CRITICAL'

        # Wait for action server asynchronously
        self.server_wait_timer = self.create_timer(
            0.15,
            self.try_connect_action_server,
            callback_group=self.cb_group
        )

    def try_connect_action_server(self):
        if not self.action_client.server_is_ready():
            self.get_logger().info('Waiting for action server waypoint_navigation...')
            return

        self.server_wait_timer.cancel()
        self.get_logger().info('Action server connected! Sending first critical goal...')
        self.send_next_goal()

    def send_next_goal(self):
        if self.goal_index >= len(self.current_goal_list):
            if self.phase == 'CRITICAL':
                self.get_logger().info('All critical goals in list completed! Triggering Switch.')
                self.trigger_switch()
            elif self.phase == 'STABLE':
                self.phase = 'COMPLETED'
                self.get_logger().info('=============================================')
                self.get_logger().info('MISSION COMPLETE! All waypoints and landing completed.')
                if self.liftoff_time:
                    total_t = time.time() - self.liftoff_time
                    self.get_logger().info(f'Total Mission Flight Time: {total_t:.2f} s')
                self.get_logger().info('=============================================')
            return

        wp = self.current_goal_list[self.goal_index]
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = float(wp['x'])
        goal_msg.waypoint.position.y = float(wp['y'])
        goal_msg.waypoint.position.z = float(wp['z'])
        goal_msg.waypoint.orientation.z = float(wp['type'])

        tag = 'SURVIVOR' if wp['type'] == 0.0 else ('PASS-THROUGH' if wp['type'] == 1.0 else 'F6 LANDING')
        self.get_logger().info(
            f"Sending {self.phase} Goal [{self.goal_index + 1}/{len(self.current_goal_list)}]: "
            f"[{wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f}] ({tag})"
        )

        send_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by server.')
            self.is_goal_active = False
            return

        self.current_goal_handle = goal_handle
        self.is_goal_active = True
        self.get_logger().info('Goal accepted by action server.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        self.is_goal_active = False
        self.current_goal_handle = None

        # GoalStatus: 4 = STATUS_SUCCEEDED, 5 = STATUS_CANCELED
        if status == 4:
            wp = self.current_goal_list[self.goal_index]
            if wp['type'] == 0.0:
                if self.phase == 'CRITICAL':
                    self.critical_saves_count += 1
                    self.get_logger().info(f'Critical survivor saved! (Total: {self.critical_saves_count}/3)')
                    if self.critical_saves_count >= 3:
                        self.get_logger().info('*** All 3 critical survivors saved! Triggering Switch immediately. ***')
                        self.trigger_switch()
                        return
                elif self.phase == 'STABLE':
                    self.stable_saves_count += 1
                    self.get_logger().info(f'Stable survivor saved! (Total: {self.stable_saves_count}/3)')

            self.goal_index += 1
            if self.phase in ('CRITICAL', 'STABLE'):
                self.send_next_goal()

        elif status == 5:
            self.get_logger().info('Goal cancellation complete on server. Proceeding to Re-plan...')
            if self.phase == 'SWITCHING':
                self.execute_replan()

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback.current_waypoint
        x = fb.pose.position.x
        y = fb.pose.position.y
        z = fb.pose.position.z

        self.latest_pos = [x, y, z]

        if not self.liftoff_detected and z < 14.6:
            self.liftoff_detected = True
            self.liftoff_time = time.time()
            self.get_logger().info('*** LIFTOFF DETECTED! Critical window timer started (t = 0) ***')

    def mission_monitor_callback(self):
        """Monitors critical window timer and triggers the Switch."""
        if self.phase != 'CRITICAL' or not self.liftoff_detected:
            return

        elapsed = time.time() - self.liftoff_time

        # Switch Condition: all 3 critical saved OR critical window (54.8s) reached
        if self.critical_saves_count >= 3 or elapsed >= 54.8:
            reason = "All 3 critical survivors saved" if self.critical_saves_count >= 3 else f"55s limit reached (t={elapsed:.1f}s)"
            self.get_logger().info(f'*** THE SWITCH TRIGGERED: {reason} ***')
            self.trigger_switch()

    def trigger_switch(self):
        if self.phase != 'CRITICAL':
            return
        self.phase = 'SWITCHING'

        # Cancel active goal in flight if present
        if self.is_goal_active and self.current_goal_handle is not None:
            self.get_logger().info('Cancelling active critical goal...')
            self.current_goal_handle.cancel_goal_async()
        else:
            self.execute_replan()

    def execute_replan(self):
        """Calls replan_route service with drone's position at the Switch."""
        self.get_logger().info(f'Calling replan_route with drone position: [{self.latest_pos[0]:.2f}, {self.latest_pos[1]:.2f}]')

        if not self.cli_replan.service_is_ready():
            self.replan_wait_timer = self.create_timer(
                0.2,
                self.try_call_replan,
                callback_group=self.cb_group
            )
            return

        self.call_replan_now()

    def try_call_replan(self):
        if not self.cli_replan.service_is_ready():
            self.get_logger().info('Waiting for replan_route service...')
            return
        self.replan_wait_timer.cancel()
        self.call_replan_now()

    def call_replan_now(self):
        req = ReplanRoute.Request()
        req.current_position.position.x = float(self.latest_pos[0])
        req.current_position.position.y = float(self.latest_pos[1])
        req.current_position.position.z = float(self.latest_pos[2])

        replan_future = self.cli_replan.call_async(req)
        replan_future.add_done_callback(self.replan_response_callback)

    def replan_response_callback(self, future):
        response = future.result()
        self.get_logger().info(f'Received {len(response.waypoints.poses)} stable route waypoints.')

        self.stable_goals = []
        for p in response.waypoints.poses:
            self.stable_goals.append({
                'x': p.position.x,
                'y': p.position.y,
                'z': p.position.z,
                'type': p.orientation.z
            })

        self.current_goal_list = self.stable_goals
        self.goal_index = 0
        self.phase = 'STABLE'

        self.get_logger().info('Starting Stable Phase! Sending first stable goal...')
        self.send_next_goal()


def main(args=None):
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_client)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
