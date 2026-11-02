#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_controller which holds the position of Swift Pico Drone.
This node publishes and subscribes the following topics:

		PUBLICATIONS			SUBSCRIPTIONS
		/drone_command			/whycode_node/markers
		/pos_error				/throttle_pid
		/controller_heartbeat	/pitch_pid
								/roll_pid
'''

import time
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header


class Swift_Pico(Node):
	def __init__(self):
		super().__init__('pico_controller')

		# [x, y, z] - WhyCode marker position
		self.current_state = [0.0, 0.0, 0.0]

		# Target setpoint: [x, y, z] = (1.5, -1.5, 13.0)
		self.desired_state = [1.5, -1.5, 13.0]

		# Command message
		self.cmd = SwiftMsgs()
		self.cmd.rc_roll = 1500
		self.cmd.rc_pitch = 1500
		self.cmd.rc_yaw = 1500
		self.cmd.rc_throttle = 1500
		self.cmd.rc_aux4 = 1500

		# PID gains: [roll, pitch, throttle]
		self.Kp = [65.0, 65.0, 60.0]
		self.Ki = [0.002, 0.002, 0.025]
		self.Kd = [95.0, 95.0, 48.0]

		# Error variables: [roll, pitch, throttle]
		self.error = [0.0, 0.0, 0.0]
		self.prev_error = [0.0, 0.0, 0.0]
		self.sum_error = [0.0, 0.0, 0.0]

		# Limits
		self.min_values = [1000, 1000, 1000]
		self.max_values = [2000, 2000, 2000]
		self.max_sum_error = [50.0, 50.0, 50.0]

		self.first_cycle = True

		# Publishers
		self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
		self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
		self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

		# Subscriptions
		self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
		self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
		self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
		self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

		# Heartbeat timer (10 Hz)
		self.publish_heartbeat()
		self.create_timer(0.1, self.publish_heartbeat)

		# Arming delay
		time.sleep(0.5)
		self.arm()

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

	def altitude_set_pid(self, alt):
		self.Kp[2] = alt.kp
		self.Ki[2] = alt.ki
		self.Kd[2] = alt.kd

	def pitch_set_pid(self, pitch):
		self.Kp[1] = pitch.kp
		self.Ki[1] = pitch.ki
		self.Kd[1] = pitch.kd

	def roll_set_pid(self, roll):
		self.Kp[0] = roll.kp
		self.Ki[0] = roll.ki
		self.Kd[0] = roll.kd

	def whycode_callback(self, msg):
		if not msg.markers:
			return

		self.current_state[0] = msg.markers[0].position.position.x
		self.current_state[1] = msg.markers[0].position.position.y
		self.current_state[2] = msg.markers[0].position.position.z

		# Error calculation:
		# pitch (x): desired_x - current_x
		# roll (y):  desired_y - current_y
		# throttle (z): current_z - desired_z (WhyCode z points downward from ceiling)
		self.error[1] = self.desired_state[0] - self.current_state[0]
		self.error[0] = self.desired_state[1] - self.current_state[1]
		self.error[2] = self.current_state[2] - self.desired_state[2]

		# Publish position errors
		pos_error = Error()
		pos_error.roll_error = float(self.error[0])
		pos_error.pitch_error = float(self.error[1])
		pos_error.throttle_error = float(self.error[2])
		pos_error.yaw_error = 0.0
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

		# Compute derivatives (WhyCode publishes at 30 Hz -> dt ≈ 0.0333s)
		dt = 0.0333
		d_roll = (self.error[0] - self.prev_error[0]) / dt
		d_pitch = (self.error[1] - self.prev_error[1]) / dt
		d_throttle = (self.error[2] - self.prev_error[2]) / dt

		# Compute PID outputs
		out_roll = (self.Kp[0] * self.error[0]
					+ self.Ki[0] * self.sum_error[0]
					+ self.Kd[0] * d_roll)

		out_pitch = (self.Kp[1] * self.error[1]
					 + self.Ki[1] * self.sum_error[1]
					 + self.Kd[1] * d_pitch)

		out_throttle = (self.Kp[2] * self.error[2]
						+ self.Ki[2] * self.sum_error[2]
						+ self.Kd[2] * d_throttle)

		# Store previous errors
		self.prev_error[0] = self.error[0]
		self.prev_error[1] = self.error[1]
		self.prev_error[2] = self.error[2]

		# Map to RC commands:
		cmd_roll = int(1500 + out_roll)
		cmd_pitch = int(1500 + out_pitch)
		cmd_throttle = int(1500 + out_throttle)

		# Clamp commands (keep throttle between 1250 and 1750 to preserve attitude authority)
		self.cmd.rc_roll = max(min(cmd_roll, 1900), 1100)
		self.cmd.rc_pitch = max(min(cmd_pitch, 1900), 1100)
		self.cmd.rc_throttle = max(min(cmd_throttle, 1750), 1250)
		self.cmd.rc_yaw = 1500
		self.cmd.rc_aux4 = 2000

		self.command_pub.publish(self.cmd)


def main(args=None):
	rclpy.init(args=args)
	swift_pico = Swift_Pico()

	try:
		rclpy.spin(swift_pico)
	except (KeyboardInterrupt, ExternalShutdownException):
		swift_pico.get_logger().info('Shutting down.\n')
	finally:
		swift_pico.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()