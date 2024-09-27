#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import argparse
import numpy as np
import yaml
import cv2

class HsvObstacleSegmentation():
    def __init__(self, robot_name, num_obstacle_points=4096*2):
        assert robot_name in {'franka', 'sawyer'}
        rospy.init_node('pc_transform')

        with open(f'config/xmop_{robot_name}.yaml') as file:
            seg_config = yaml.safe_load(file)['obstacle_seg']

        self.frame_id = seg_config['frame_id']
        np_dtype_dict = dict(
            realsense=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('rgb', '<f4')],
            zed2=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')],
        )
        self.np_dtype = np_dtype_dict[seg_config['camera']]
        
        self.num_obstacle_points = num_obstacle_points
        self.low_h, self.high_h, self.low_s, self.high_s, self.low_v, self.high_v = seg_config['hsv_thresh']
            
        # based on camera extrincs (changes with calibration)
        self.T_base_depth_optical = np.array(seg_config['pc_transform'])

        self.pc_sub = rospy.Subscriber(seg_config['pc_topic'], PointCloud2, self.pc_callback, queue_size=1)
        self.pc_pub = rospy.Publisher('/xmop/obstacle_points', PointCloud2, queue_size=1)
        rospy.sleep(1) # wait for subscriptions

    def pc_callback(self, msg):
        np_pc = np.frombuffer(msg.data, dtype=self.np_dtype)
        points = np.hstack((np.expand_dims(np_pc['x'],-1), np.expand_dims(np_pc['y'], -1), np.expand_dims(np_pc['z'],-1)))
        points = points.reshape(msg.height,msg.width,3)
        rgb = np.frombuffer(np.ascontiguousarray(np_pc['rgb']).data, dtype=np.uint8)
        rgb = rgb.reshape(msg.height*msg.width,4)[:,:3].reshape(msg.height,msg.width,3)

        # set this ros parameter to true for showing the hsv threshold sliders 
        # calibrate to the experiment lighting conditions
        # set the ros parameter to false 
        if rospy.get_param('hsv_tuning', False):
            cv2.imshow('rgb', rgb)
            cv2.waitKey(1)

            cv2.namedWindow('trackbars')
            def callback(x):
                pass
            cv2.createTrackbar('Low H', 'trackbars', self.low_h, 179, callback)
            cv2.createTrackbar('High H', 'trackbars', self.high_h, 179, callback)
            cv2.createTrackbar('Low S', 'trackbars', self.low_s, 255, callback)
            cv2.createTrackbar('High S', 'trackbars', self.high_s, 255, callback)
            cv2.createTrackbar('Low V', 'trackbars', self.low_v, 255, callback)
            cv2.createTrackbar('High V', 'trackbars', self.high_v, 255, callback)

            while rospy.get_param('hsv_tuning', False):
                self.low_h = cv2.getTrackbarPos('Low H', 'trackbars')
                self.high_h = cv2.getTrackbarPos('High H', 'trackbars')
                self.low_s = cv2.getTrackbarPos('Low S', 'trackbars')
                self.high_s = cv2.getTrackbarPos('High S', 'trackbars')
                self.low_v = cv2.getTrackbarPos('Low V', 'trackbars')
                self.high_v = cv2.getTrackbarPos('High V', 'trackbars')

                lower_hsv = np.array([self.low_h, self.low_s, self.low_v])
                upper_hsv = np.array([self.high_h, self.high_s, self.high_v])
                hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)                
                
                cv2.imshow('mask', mask)
                cv2.waitKey(1)

        lower_hsv = np.array([self.low_h, self.low_s, self.low_v])
        upper_hsv = np.array([self.high_h, self.high_s, self.high_v])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv) # only works for monocolor obstacles
        mask_indices = mask==255

        if np.any(mask_indices):
            local_points = points[mask_indices].reshape(-1,3)
            local_points = local_points[np.random.choice(local_points.shape[0], self.num_obstacle_points, replace=True)]
            local_points = np.concatenate((local_points, np.ones((local_points.shape[0],1))), axis=-1)
            global_points = (self.T_base_depth_optical@local_points.T).T[:,:3] # use calibration matrix to transform pc to robot's base
            global_points = global_points[global_points[:,0] > 0.3] # crop the pc to remove unwanted points
            global_points = global_points[global_points[:,0] < 1.5]
            
            self.pc_pub.publish(self.numpy_to_rosmsg(global_points.astype(np.float32), self.frame_id))

    def numpy_to_rosmsg(self, points, parent_frame, stamp=None):
        dtype_float32 = 4
        ros_dtype_float32 = PointField.FLOAT32
        data = points.tobytes()
        fields = [PointField(name='x', offset=0*dtype_float32, datatype=ros_dtype_float32, count=1),
                    PointField(name='y', offset=1*dtype_float32, datatype=ros_dtype_float32, count=1),
                    PointField(name='z', offset=2*dtype_float32, datatype=ros_dtype_float32, count=1)]
        point_step = 3*dtype_float32
        
        if stamp == None:
            stamp = rospy.Time.now()
        header = Header(frame_id=parent_frame, stamp=stamp)

        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=point_step,
            row_step=(point_step * points.shape[0]),
            data=data
        ) # unordered pointcloud


# rosparam set hsv_tuning true
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XMoP-Interface')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot under the context')
    parser.add_argument('--num_obs_pts', default=4096*4, type=str, help='Number of obstacle surface points')
    args = parser.parse_args()
    print(args)

    node = HsvObstacleSegmentation(args.robot_name, args.num_obs_pts)
    rospy.spin()