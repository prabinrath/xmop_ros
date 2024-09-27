xhost +
docker run --rm -it --gpus all --network=host --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $1:/root/catkin_ws/src/xmop_ros -v $2:/root/xmop prabinrath/xmop_deploy:latest bash
