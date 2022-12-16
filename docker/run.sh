#!/bin/bash

cd `dirname $0`

# xhost +
xhost +local:user
# xhost + 192.168.1.54
nvidia-docker &> /dev/null
if [ $? -ne 0 ]; then
    echo $TAG
    echo "=============================" 
    echo "=nvidia docker not installed="
    echo "============================="
else
    echo "=========================" 
    echo "=nvidia docker installed="
    echo "========================="
    docker run -it \
    --privileged \
    --runtime=nvidia \
    --env=DISPLAY=$DISPLAY \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/home/${USER}/.Xauthority:/home/${USER}/.Xauthority" \
    --env="QT_X11_NO_MITSHM=1" \
    --rm \
    -v "/$(pwd)/global_ros_setting.sh:/ros_setting.sh" \
    -v "/$(pwd)/ros_workspace:/home/${USER}/catkin_ws/" \
    -v "/$(pwd)/../whill_path:/home/${USER}/catkin_ws/src/whill_path" \
    -v "/$(pwd)/../third_party:/home/${USER}/catkin_ws/src/third_party" \
    -v "${PWD}/config/terminator_config:/home/${USER}/.config/terminator/config" \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /media:/media \
    -v /dev:/dev \
    -v /mnt/ssd:/mnt/ssd \
    --net host \
    ${USER}/whill_path
fi

