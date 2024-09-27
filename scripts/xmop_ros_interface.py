#!/usr/bin/env python
from xmop import XMoP
from common.robot_point_sampler import RealRobotPointSampler
from sensor_msgs.msg import JointState, PointCloud2
from actionlib import SimpleActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, \
                             FollowJointTrajectoryActionGoal, \
                             FollowJointTrajectoryGoal, \
                             FollowJointTrajectoryResult
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from threading import Lock, Thread
from scipy.interpolate import CubicSpline
from copy import deepcopy
import numpy as np
import rospy
import yaml


class XMoPROSInterface():
    def __init__(self, 
                 topics_config,
                 mode='offline',
                 robot_name='franka', 
                 algo='xmop-s', 
                 exec_freq=5,
                 query_factor=0.01,
                 smoothing_factor=0.5,
                 goal_thresh=0.05,
                 max_rollout_steps=1000,
                 joint_state_mask=[True,]*7,
                 device='cuda'):
        # Planner configuration
        assert algo in {'xmop-s', 'xmop'}
        assert robot_name in {'franka', 'sawyer'}
        assert mode in {'online', 'offline'}
        self.robot_name = robot_name
        self.mode = mode
        self.device = device
        self.exec_freq = exec_freq
        self.query_factor = query_factor
        self.goal_thresh = goal_thresh
        self.max_rollout_steps = max_rollout_steps
        self.joint_state_mask = np.asarray(joint_state_mask)

        urdf_path = {
            'franka': '/root/xmop/urdf/franka_panda/panda_sample.urdf',
            'sawyer': '/root/xmop/urdf/sawyer/sawyer_sample.urdf'
            # Add your new robot URDF here
        }

        with open('/root/xmop/config/robot_point_sampler.yaml') as file:
            robot_point_sampler = RealRobotPointSampler(
                urdf_path=urdf_path[robot_name], 
                config=yaml.safe_load(file)['xmop_planning'],
                device=device)
            self.home_config = robot_point_sampler.home_config

        planner_config = dict(
            mode='singlestep' if algo == 'xmop-s' else 'multistep',
            urdf_path=urdf_path[robot_name],
            config_folder='/root/xmop/config/',
            model_dir=None,
            smoothing_factor=smoothing_factor
        )
        self.nx_planner = XMoP(planner_config,
                            robot_point_sampler=robot_point_sampler,
                            device=device)
    
        # ROS configuration
        self.js_lock = Lock()
        self.joint_state = None
        self.joint_names = None
        self.js_subscriber = rospy.Subscriber(
            topics_config['js_topic'],
            JointState,
            self.js_callback,
            queue_size=1,
        )

        self.pc_lock = Lock()
        self.pointcloud = None
        self.pc_subscriber = rospy.Subscriber(
            topics_config['pc_topic'],
            PointCloud2,
            self.pc_callback,
            queue_size=1,
        )        

        # action client for open-loop control with feedback 
        self.offline_action_client = SimpleActionClient(topics_config['offline_action_topic'], FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for robot action server')
        self.offline_action_client.wait_for_server()

        self.rollout_rviz_publisher = rospy.Publisher(
            topics_config['rollout_rviz_topic'],
            DisplayTrajectory,
            queue_size=1,
        )

        self.online_command_publisher = rospy.Publisher(
            topics_config['online_command_topic'],
            FollowJointTrajectoryActionGoal,
            queue_size=1,
        )

        # state machine setup
        self.planned_traj = None
        self.goal_eef = None
        self.planning_lock = Lock()

        self.PLAN = False
        self.EXECUTE = False
        self.ONLINE = False
        self.STOP = False
        exec_thread = Thread(target=self.loop)
        exec_thread.start()
        rospy.sleep(1) # wait for subscriptions
    
    def js_callback(self, msg):
        if self.joint_names is None:
            try:
                self.joint_names = []
                for j, mask in enumerate(self.joint_state_mask):
                    j_name = msg.name[j]
                    if mask:
                        self.joint_names.append(j_name)
            except:
                rospy.logwarn('Joint mask mismatch')
                self.joint_names = None

        if self.joint_names is not None:
            try:
                joint_state = []
                for j, mask in enumerate(self.joint_state_mask):
                    j_pos = msg.position[j]
                    if mask:
                        joint_state.append(j_pos)
                with self.js_lock:
                    self.joint_state = np.asarray(joint_state)
            except:
                pass

    def pc_callback(self, msg):
        np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        np_pc = np.frombuffer(msg.data, dtype=np_dtype)
        points = np.hstack((np.expand_dims(np_pc['x'],-1), np.expand_dims(np_pc['y'], -1), np.expand_dims(np_pc['z'],-1)))
        points = np.concatenate((points, np.zeros((points.shape[0],1))), axis=-1)
        with self.pc_lock:
            self.pointcloud = points[np.random.choice(points.shape[0], 4096*2, replace=True)]
    
    def set_goal(self, goal_eef):
        acq_planning = self.planning_lock.acquire(timeout=1)
        if acq_planning:
            self.goal_eef = goal_eef
            self.planning_lock.release()
        elif not acq_planning:
            rospy.logwarn('Currently planning for an existing goal')

    def plan_online(self):
        observation = {}
        with self.js_lock:
            observation['curr_js'] = np.copy(self.joint_state)
        with self.pc_lock:
            observation['obstacle_surface_pts'] = np.copy(self.pointcloud)

        data_dict_batch = self.nx_planner.prepare_data_dict_batch(self.goal_eef)
        self.planned_traj, reached_goal = self.nx_planner.plan_online(observation, 
                                                                    data_dict_batch,
                                                                    self.goal_thresh)
        return reached_goal
    
    def postprocess_trajectory(self, traj, query_factor=0.01, k_size=4):
        # fit a fine cubic spline for interpolation
        dt = 1/self.exec_freq
        fit_array_x = np.arange(traj.shape[0]) * dt
        fit_array_x_query = np.linspace(0, fit_array_x[-1], int(fit_array_x[-1]/(dt*query_factor)))
        cs = CubicSpline(fit_array_x, traj)
        spline_traj = np.array([cs(x) for x in fit_array_x_query])

        # smoothen the trajectory for execution
        smoothened_traj = np.copy(spline_traj)
        kernel = np.ones(k_size)/k_size
        for dim in range(spline_traj.shape[1]):
            smoothened_traj[k_size//2:-(k_size//2)+(1-k_size%2), dim] = \
                np.convolve(spline_traj[:,dim], kernel, 'valid')
        
        return fit_array_x_query, smoothened_traj


    def plan_offline(self, max_rollout_steps):
        observation = {}
        with self.js_lock:
            observation['curr_js'] = np.copy(self.joint_state)
        with self.pc_lock:
            observation['obstacle_surface_pts'] = np.copy(self.pointcloud)

        with self.planning_lock:
            observation['goal_eef'] = self.goal_eef
             
            self.planned_traj = self.nx_planner.plan_offline(observation, 
                            int(max_rollout_steps/(1-self.nx_planner.smoothing_factor)), 
                            self.goal_thresh, 
                            exact=True)
            
            if self.planned_traj is not None:
                if self.planned_traj.shape[0] == 0:
                    rospy.loginfo('Already at goal')
                    return
                rospy.loginfo('Planning successful')
                self.planned_traj = np.concatenate((np.expand_dims(observation['curr_js'], 0), self.planned_traj))
                msg = DisplayTrajectory()
                msg.trajectory_start.joint_state = JointState(position=observation['curr_js'])
                trajectory = RobotTrajectory()
                trajectory.joint_trajectory.joint_names = self.joint_names
                start_times, joint_states = np.arange(self.planned_traj.shape[0]) * (1/self.exec_freq), self.planned_traj
                for st, js in zip(start_times, joint_states):
                    trajectory.joint_trajectory.points.append(
                        JointTrajectoryPoint(positions=js,
                                             time_from_start=rospy.Duration.from_sec(st))
                    )
                msg.trajectory.append(trajectory)
                self.rollout_rviz_publisher.publish(msg)
            else:
                rospy.logwarn('Planning failed')
    
    def execute_offline(self, start_times, joint_states):
        goal_traj = FollowJointTrajectoryGoal()
        goal_traj.trajectory.joint_names = self.joint_names
        for st, js in zip(start_times, joint_states):
            point = JointTrajectoryPoint()
            point.positions = js
            point.time_from_start = rospy.Duration.from_sec(st)
            goal_traj.trajectory.points.append(point)

        goal_traj.goal_time_tolerance = rospy.Duration.from_sec(1.0)
        # close-loop execution
        self.offline_action_client.send_goal_and_wait(goal_traj)
        result = self.offline_action_client.get_result()
        if result.error_code != FollowJointTrajectoryResult.SUCCESSFUL:
            rospy.logerr('XMoP: Movement was not successful: ' + {
                FollowJointTrajectoryResult.INVALID_GOAL:
                """
                The joint pose you want to move to is invalid (e.g. unreachable, singularity...).
                Is the 'joint_pose' reachable?
                """,

                FollowJointTrajectoryResult.INVALID_JOINTS:
                """
                The joint pose you specified is for different joints than the joint trajectory controller
                is claiming. Does you 'joint_pose' include all 7 joints of the robot?
                """,

                FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED:
                """
                During the motion the robot deviated from the planned path too much. Is something blocking
                the robot?
                """,

                FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED:
                """
                After the motion the robot deviated from the desired goal pose too much. Probably the robot
                didn't reach the joint_pose properly
                """,
            }[result.error_code])
    
    def reset_robot(self):
        with self.js_lock:
            curr_js = np.copy(self.joint_state)
        max_movement = np.max(abs(self.home_config-curr_js))
        start_time = max(max_movement / 0.1, 0.5)
        self.execute_offline([start_time], [self.home_config])
    
    def loop(self):
        while not self.STOP:
            if self.PLAN:
                self.plan_offline(max_rollout_steps=self.max_rollout_steps)
                self.PLAN = False

            if self.EXECUTE:
                if self.mode == 'online':
                    # closed-loop execution
                    local_traj = None
                    with self.planning_lock:
                        reached_goal = self.plan_online()
                        local_traj = deepcopy(self.planned_traj)

                    if self.robot_name == 'sawyer':
                        # hack for sawyer as the robot is too jerky
                        local_traj = local_traj[:2]

                    if local_traj is not None:
                        for js in local_traj:
                            next_js_goal = FollowJointTrajectoryActionGoal()
                            next_js_goal.goal.trajectory.joint_names = self.joint_names

                            if self.robot_name == 'sawyer':
                                # hack for sawyer as the action server does not take single waypoint trajectory
                                with self.js_lock:
                                    curr_js = self.joint_state
                                point = JointTrajectoryPoint(positions=curr_js)
                                point.time_from_start = rospy.Duration.from_sec(0)
                                next_js_goal.goal.trajectory.points.append(point)

                            point = JointTrajectoryPoint(positions=js)
                            point.time_from_start = rospy.Duration.from_sec(1/self.exec_freq)
                            next_js_goal.goal.trajectory.points.append(point)
                            next_js_goal.goal.goal_time_tolerance = rospy.Duration.from_sec(0.5)
                            self.online_command_publisher.publish(next_js_goal)
                            rospy.sleep(1/self.exec_freq)
                        if reached_goal:
                            rospy.loginfo('Reached gaol stopping now')
                        self.EXECUTE = not reached_goal and self.ONLINE
                else:
                    with self.planning_lock:
                        complete_traj = deepcopy(self.planned_traj)
                    
                    if complete_traj is not None:
                        if complete_traj.shape[0] > 0:
                            start_times, joint_states = self.postprocess_trajectory(self.planned_traj, 
                                                                                    query_factor=self.query_factor)
                            # open-loop execution
                            self.execute_offline(start_times, joint_states)
                            with self.planning_lock:
                                self.planned_traj = None
                        else:
                            rospy.logwarn('Plan not ready yet')
                    else:
                        rospy.logwarn('Plan not ready yet')
                    self.EXECUTE = False     
        