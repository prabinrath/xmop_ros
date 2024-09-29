# ROS Package for [XMoP](https://prabinrath.github.io/xmop/)
![ROS: Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg) <br>
XMoP is a novel configuration-space neural policy that solves motion planning problems zero-shot for unseen robotic manipulators. This repository contains the ROS package for sim-to-real deployment of XMoP neural planner. The official implementation repository can be found [here](https://github.com/prabinrath/xmop). We provide a docker container pre-configured with ROS, PyTorch, and other dependencies to aid the deployment. <br>
<div align="center">
  <img src="./rviz_demo.gif" alt="rviz demo">
</div>

## Usage
1. Clone [xmop](https://github.com/prabinrath/xmop), and [xmop_ros](https://github.com/prabinrath/xmop_ros) to your local.
2. The XMoP ROS package should be run on a workstation with GPU setup for inference.
3. The action server for the robot needs to be set up by the user before using XMoP. For example, see the [sawyer_robot](https://github.com/RethinkRobotics/sawyer_robot) package for the Sawyer robot and [franka_ros](https://github.com/frankaemika/franka_ros) package for the Panda robot.
4. Modify the robot-specific config file in the `config/` folder to set up the ROS topics and other essential parameters.
5. A depth camera needs to be set up and calibrated by the user such that the environment pointcloud is available on the ROS server. We have provided `scripts/generate_transforms_<camera-name>.py`, `scripts/obstacle_segmentation_hsv.py`, and `launch/rollout_station.launch` for reference. However, we believe there are various ways to extract obstacle pointclouds that the user might prefer to integrate with XMoP. As long as the pointcloud does not include points corresponding to the robot itself, it can be used by XMoP for motion planning. We encourage exploring other avenues such as [realtime filters](https://github.com/blodow/realtime_urdf_filter), and [SAM2](https://github.com/facebookresearch/segment-anything-2).
6. On the workstation system, install docker from this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04). Install `nvidia-container-toolkit` from this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
7. Run the docker container.
```
cd <xmop_ros-root-directory>
bash xmop_deploy_docker.sh ./ <xmop-root-directory>
```
8. Build ROS package.
```
cd root/catkin_ws
catkin build
```
9. Setup the `ROS_MASTER_URI` and `ROS_IP` variables in `/root/.bashrc` file.
10. Exec into the container from a new terminal and run the following commands to spin up the RViz window. The UI has been borrowed from [MpiNets](https://github.com/NVlabs/motion-policy-networks?tab=readme-ov-file#interactive-demo-using-ros), please watch their demo for reference.
```
source /root/catkin_ws/devel/setup.bash
roslaunch xmop_ros rollout_commander.launch <robot-name>
```
11. Exec into the container from a new terminal and run the following commands to start XMoP planner node. Use the `-h` flag for parameters.
```
conda activate xmop_deploy
cd catkin_ws/src/xmop_ros
python scripts/xmop_ros_interaction.py
```
You can interact with the floating end-effector through RViz to set different planning goals. Once the goal is set, click on the `Plan` and `Execute` buttons to run the XMoP planner. If you find this codebase useful in your research, please cite [the XMoP paper](https://arxiv.org/pdf/2409.15585):
```bibtex
@article{rath2024xmop,
      title={XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning}, 
      author={Prabin Kumar Rath and Nakul Gopalan},
      year={2024},
      eprint={2409.15585},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.15585}, 
}
```
