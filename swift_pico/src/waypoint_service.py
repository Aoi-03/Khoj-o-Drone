#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints

class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')
        #create a service for the service type 'GetWaypoints' with the name 'waypoints' and self.waypoint_callback as the callback.
        #Refer to Writing a simple service and client (Python) in ROS 2 tutorials
        self.waypoints = []

    
    def waypoint_callback(self, request, response):

        if request.get_waypoints == True :
            response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]
            for i in range(len(self.waypoints)):
                response.waypoints.poses[i].position.x = self.waypoints[i][0]
                response.waypoints.poses[i].position.y = self.waypoints[i][1]
                response.waypoints.poses[i].position.z = self.waypoints[i][2]
            self.get_logger().info("Incoming request for Waypoints")
            return response

        else:
            self.get_logger().info("Request rejected")

def main():
    rclpy.init()
    waypoints = WayPoints()

    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        

        