from pyquaternion import Quaternion
import numpy as np

# camera intrinsics for Intel Realsense D435 (fixed)
T_link_color_optical = np.array([[ 0.00200796, -0.00199196,  0.999996  , -0.        ],
       [-0.999966  , -0.00800186,  0.00199196,  0.015     ],
       [ 0.00799786, -0.999966  , -0.00200796,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
T_link_depth_optical = np.array([[ 0.,  0.,  1.,  0.],
       [-1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]])

# camera extrinsics from Hand-to-Eye calibration (changes)
T_base_color_optical = np.zeros((4,4))
T_base_color_optical[3,3] = 1
calib_text = "1.31674 -0.00840855 0.82884   0.633083 0.649086 -0.31144 -0.284426"
calib_np = []
for val in calib_text.split(' '):
    try:
        val_num = float(val)
        calib_np.append(val)
    except:
        pass
calib_np = np.array(calib_np)
T_base_color_optical[:3,3] = calib_np[:3]
T_base_color_optical[:3,:3] = Quaternion(calib_np[6], calib_np[3], calib_np[4], calib_np[5]).rotation_matrix

T_base_link = T_base_color_optical @ np.linalg.inv(T_link_color_optical)
calib_np = np.concatenate((T_base_link[:3,3], Quaternion(matrix=T_base_link[:3,:3]).q[[1,2,3,0]]))
calib_text = ""
for val_num in calib_np:
    calib_text += f'{round(val_num, 8)} '
print(calib_text)

T_base_depth_optical = T_base_link @ T_link_depth_optical
# print(T_base_depth_optical[:3,3], Quaternion(matrix=T_base_depth_optical[:3,:3]).q[[1,2,3,0]])
print(T_base_depth_optical.tolist())
