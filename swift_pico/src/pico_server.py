#!/usr/bin/env python3

# This python file runs a ROS 2-node of name waypoint_server which implements an action server to navigate the Swift Pico Drone to the given waypoints.
# You can use either PID or LQR controller to navigate the drone to the given waypoints.


import time
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

#import control specific libraries

#import the action

#pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from std_msgs.msg import Header
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from controller_msg.msg import PIDTune
from nav_msgs.msg import Odometry

class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_or_lqr_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.odometry_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0


        self.yaw = 0.0              # current yaw in degrees (updated from whycode)
        self.desired_yaw = -0.5     # target yaw angle (degrees)
        self.curr_state = [0.0, 0.0, 0.0]
        self.desired_state = [0.0, 0.0, 0.0]
        self.drone_position = [0.0, 0.0, 0.0]
        self.xyz = [0.0, 0.0, 0.0, 0.0]
        self.dtime = 0

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1400

        #Initiate or declare other variables here depending upon whether you are implementing PID or LQR controller.



        self.sample_time = 0.01666 #put the appropriate value according to your controller

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        #Add other subscribers here

        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10, callback_group=self.odometry_callback_group)

        #create an action server for the action 'NavToWaypoint'. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        #action name should 'waypoint_navigation'.
        #include the action_callback_group in the action server. Refer to executors in ROS 2 concepts

        
        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)  # let the heartbeat reach the rosbag before arming
        self.arm()

        #define the function to be run inside the timer callback. This function will implement the PID or LQR algorithm
        self.timer = self.create_timer(self.sample_time, self., callback_group=self.pid_or_lqr_callback_group)

    def publish_heartbeat(self):
        hb = Header()
        hb.stamp = self.get_clock().now().to_msg()
        hb.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(hb)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)


    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)


    # Whycode callback function
    # The function gets executed each time whycode_node publishes its markers.
    def whycode_callback(self, msg):

        if not msg.markers:
            self.get_logger().warn("No markers received")
            return

        marker = msg.markers[0]
        position = marker.position.position
        self.curr_state = [position.x, position.y, position.z]
        self.drone_position = self.curr_state[:]

        # the yaw controller works in degrees, so convert here.
        self.yaw = math.degrees(marker.rotation.z)

        self.dtime = msg.header.stamp.sec

    # If you are using PID controller, then define callback function like altitide_set_pid to tune pitch, roll.
    #If you are using LQR controller, then define a functions which were given in the boiler plate code of task 1c and the ones which you have defined on yur own.


    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)

    #define the function to be run inside the timer callback. This function will implement the PID or LQR algorithm. 
    # This will be either 'pid' finction or 'controller' function as you can see in the boiler plate for PID or LQR for task 1c.
    # You can also add yaw control here (P, PD or LQR) to hold self.desired_yaw using self.yaw (in degrees, from whycode) and set self.cmd.rc_yaw.






    def execute_callback(self, goal_handle):

        self.get_logger().info('Executing goal...')
        self.desired_state[0] = goal_handle.request.waypoint.position.x
        self.desired_state[1] = goal_handle.request.waypoint.position.y
        self.desired_state[2] = goal_handle.request.waypoint.position.z

        self.get_logger().info(f'New Waypoint Set: {self.desired_state}')
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        #create a NavToWaypoint feedback object. Refer to Writing an action server and client (Python) in ROS 2 tutorials.
        
        #--------The script given below checks whether you are hovering at each of the waypoints(goals) for max of 3s---------#
        # This will help you to analyse the drone behaviour and help you to tune the PID better.

        while True:
			#current whycon poses
            feedback_msg.current_waypoint.pose.position.x = self.curr_state[0]
            feedback_msg.current_waypoint.pose.position.y = self.curr_state[1]
            feedback_msg.current_waypoint.pose.position.z = self.curr_state[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)

            drone_is_in_sphere = self.is_drone_in_sphere(self.drone_position, goal_handle, 0.08) 

            # The value 0.08 is in meters used for as error range in Whycon coordinates for LQR controller. If you are using PID controller, it will become 0.8 as PID Whycon coordinates are in decimeters.

            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        pass
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        self.point_in_sphere_start_time = self.dtime
                        self.get_logger().info('Drone in sphere for 1st time')                        #you can choose to comment this out to get a better look at other logs

            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                        self.get_logger().info('Drone in sphere')                                     #you can choose to comment this out to get a better look at other logs
                             
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.get_logger().info('Drone out of sphere')                                 #you can choose to comment this out to get a better look at other logs
                        self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                 self.max_time_inside_sphere = self.time_inside_sphere

            if self.max_time_inside_sphere >= 3:
                 break
                        

        goal_handle.succeed()

        #create a NavToWaypoint result object. Refer to Writing an action server and client (Python) in ROS 2 tutorials

        result.hov_time = self.dtime - self.duration #this is the total time taken by the drone in trying to stabilize at a point
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2


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
         rclpy.shutdown()


if __name__ == '__main__':
    main()

