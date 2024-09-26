from pyquaternion import Quaternion
import numpy as np

# camera intrinsics for Zed (fixed)
T_link_color_optical = np.array([[ 0.,  0.,  1.,  0.],
       [-1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]])

# camera extrinsics from Hand-to-Eye calibration (changes)
T_base_color_optical = np.zeros((4,4))
T_base_color_optical[3,3] = 1
calib_text = "1.01393 -0.56424 0.668037   0.90263 0.256361 -0.0968864 -0.331891"
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

T_base_depth_optical = np.copy(T_base_link)
print(T_base_depth_optical[:3,3], Quaternion(matrix=T_base_depth_optical[:3,:3]).q[[1,2,3,0]])
print(T_base_depth_optical.tolist())
