#!/usr/bin/env python
import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    Marker,
    InteractiveMarkerFeedback,
    InteractiveMarker,
    InteractiveMarkerControl,
)
from xmop_ros_interface import XMoPROSInterface
import argparse
from copy import deepcopy
import numpy as np
import yaml


class XMoPROSInteraction:
    def __init__(self, config):
        """
        Initialize the system state, the interactive components, and the subscribers/publishers
        """

        rospy.init_node("xmop_interface")
        self.robot_name = config['robot_name']
        self.mode = config['mode']
        self.base_frame_id = config['base_frame_id']

        self.xmop_interface = XMoPROSInterface(config['topics_config'],
                                               mode=self.mode,
                                               robot_name=self.robot_name,
                                               algo=config['algo'],
                                               exec_freq=config['exec_freq'],
                                               query_factor=config['query_factor'],
                                               max_rollout_steps=config['max_rollout_steps'],
                                               smoothing_factor=config['smoothing_factor'],
                                               joint_state_mask=config['joint_state_mask'],
                                               device='cuda')

        self.server = InteractiveMarkerServer("xmop_controls", "")
        self.make_reset_button_marker([0.4, -1.0, 0.1], 0.2)
        self.make_execute_button_marker([1.0, -1.0, 0.1], 0.2)
        if  self.mode=='offline':
            self.make_plan_button_marker([0.7, -1.0, 0.1], 0.2)
        
        self.goal_xyz, self.goal_quat_xyzw = config['default_goal']
        self.make_goal_marker(
            self.goal_xyz,
            self.goal_quat_xyzw,
        )
        self.server.applyChanges()

        if self.mode == 'online':
            self.xmop_interface.set_goal((np.array(self.goal_xyz),
                                          np.array(self.goal_quat_xyzw)[[3,0,1,2]]))

    def make_box(self, side_length, color):
        """
        Makes a colored box that can be viewed in Rviz (will be used as buttons)
        """
        marker = Marker()
        marker.type = Marker.CUBE

        marker.scale.x = side_length
        marker.scale.y = side_length
        marker.scale.z = side_length
        (
            marker.color.r,
            marker.color.g,
            marker.color.b,
            marker.color.a,
        ) = color
        return marker
    
    def make_gripper(self):
        """
        Creates a floating gripper that can be viewed in Rviz (will be used as the goal)
        """
        marker = Marker()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = f"package://xmop_ros/meshes/{self.robot_name}_gripper.stl"

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        return marker

    def make_reset_button_marker(self, xyz, side_length):
        """
        Creates a red cube that resets the system when you click on it
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame_id
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "reset_button"
        int_marker.description = "Reset"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "reset_button_control"

        marker = self.make_box(side_length, [204.0 / 255, 50.0 / 255, 50.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.reset_button_callback)
    
    def reset_button_callback(self, feedback):
        """
        A callback that's called after clicking on the reset button, resets the system
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Resetting robot to home state")
            self.xmop_interface.reset_robot()
        self.server.applyChanges()

    def make_plan_button_marker(self, xyz, side_length):
        """
        Create a yellow cube that calls the planner and visualizes the result when you click on it
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame_id
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "plan_button"
        int_marker.description = "Plan"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "plan_button_control"

        marker = self.make_box(side_length, [231.0 / 255, 180.0 / 255, 22.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.plan_button_callback)
    
    def plan_button_callback(self, feedback):
        """
        This is called whenever the plan button is clicked.
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Sending planning pose goal")
            self.xmop_interface.set_goal((np.array(self.goal_xyz),
                                          np.array(self.goal_quat_xyzw)[[3,0,1,2]]))
            self.xmop_interface.PLAN = True
        self.server.applyChanges()

    def make_execute_button_marker(self, xyz, side_length):
        """
        Create a green cube button that executes on the robot when you click on it
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame_id
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "execute_button"
        int_marker.description = "Execute"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "execute_button_control"

        marker = self.make_box(side_length, [45.0 / 255, 201.0 / 255, 55.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.execute_button_callback)
    
    def execute_button_callback(self, feedback):
        """
        This is called whenever the execute button is clicked.
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            if not self.xmop_interface.EXECUTE:
                rospy.logwarn("Executing plan")
                self.xmop_interface.EXECUTE = True
                self.xmop_interface.ONLINE = True
            else:
                rospy.logwarn("Stopping plan")
                self.xmop_interface.EXECUTE = False
                self.xmop_interface.ONLINE = False
        self.server.applyChanges()

    def make_goal_marker(self, xyz, quat):
        """
        Create the goal interactive marker
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame_id
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        (
            int_marker.pose.orientation.x,
            int_marker.pose.orientation.y,
            int_marker.pose.orientation.z,
            int_marker.pose.orientation.w,
        ) = quat
        int_marker.scale = 0.4

        int_marker.name = "goal"
        int_marker.description = "Goal"

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.make_gripper())
        control.interaction_mode = InteractiveMarkerControl.NONE
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()

        quat = np.array([1.0, 0.0, 0.0, 1.0])
        quat = quat / np.linalg.norm(quat)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = quat
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        quat = np.array([0.0, 1.0, 0.0, 1.0])
        quat = quat / np.linalg.norm(quat)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = quat
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        quat = np.array([0.0, 0.0, 1.0, 1.0])
        quat = quat / np.linalg.norm(quat)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = quat
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.goal_feedback)

    def goal_feedback(self, feedback):
        """
        This is called whenever the user interacts with the goal marker. This is used to
        set the goal pose.
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.goal_xyz = (
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            )
            self.goal_quat_xyzw = (
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            )

            if self.mode == 'online':
                self.xmop_interface.set_goal((np.array(self.goal_xyz),
                                            np.array(self.goal_quat_xyzw)[[3,0,1,2]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XMoP-Interface')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot under the context')
    parser.add_argument('--algo', default='xmop', type=str, help='Planning algorithm to use')
    parser.add_argument('--mode', default='offline', type=str, help='Execution mode')
    args = parser.parse_args()
    print(args)

    with open(f'config/xmop_{args.robot_name}.yaml') as file:
        robot_policy_config = yaml.safe_load(file)
        algo_config = robot_policy_config[f'{args.algo}-{args.mode}']
    
    xmop_config = dict(
        mode=args.mode,
        robot_name=args.robot_name,
        topics_config=robot_policy_config['topics_config'],
        algo=args.algo,
        exec_freq=algo_config['exec_freq'],
        query_factor=algo_config['query_factor'],
        smoothing_factor=algo_config['smoothing_factor'],
        max_rollout_steps=algo_config['max_rollout_steps'],
        default_goal=robot_policy_config['default_goal'],
        base_frame_id = robot_policy_config['obstacle_seg']['frame_id'],
        joint_state_mask=robot_policy_config['joint_state_mask']
    )
    node = XMoPROSInteraction(xmop_config)
    rospy.spin()