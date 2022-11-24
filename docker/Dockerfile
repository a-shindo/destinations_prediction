# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images/create_ros_core_image.Dockerfile.em
FROM ubuntu:20.04
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID=${UID}
ARG SHELL=/bin/bash
#FROM $base_image
RUN echo base image: ${base_image}

#######################################################################
##                            Speeding up                            ##
#######################################################################
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

#######################################################################
##                      install common packages                      ##
#######################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
   pkg-config \
   apt-utils \
   wget \
   git \
   build-essential \ 
   net-tools \
   gedit \
   terminator \
   nautilus \
   software-properties-common \
   apt-transport-https \
   libopencv-dev \
   ffmpeg \
   x264 \
   libx264-dev \
   zip \
   unzip \
   usbutils \
   sudo \
   python3-pip \
   libusb-1.0-0-dev \
   dbus-x11

#######################################################################
##                           install font                            ##
#######################################################################
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections 
RUN apt-get update && apt-get install -y ttf-mscorefonts-installer \
    ttf-ubuntu-font-family \
    msttcorefonts -qq

RUN python3 -m pip install --upgrade pip
RUN apt-get install  -y python3-ruamel.yaml

#######################################################################
##                       install nvidia docker                       ##
#######################################################################
RUN apt-get install -y --no-install-recommends \
    libxau-dev \
    libxdmcp-dev \
    libxcb1-dev \
    libxext-dev \
    libx11-dev \
    mesa-utils \
    x11-apps

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64


#######################################################################
##                            ros install                            ##
#######################################################################

# install packages
RUN apt-get update
RUN apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    curl

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update 
RUN apt-get install -y \
    ros-noetic-desktop-full
 # install ros packages
ENV ROS_DISTRO noetic
COPY ./ros_entrypoint.sh /

RUN apt-get install -y \
    python3-rosdep \
    python3-rosinstall \ 
    python3-rosinstall-generator \ 
    python3-wstool \
    build-essential 
#######################################################################
##                   install additional packages                     ##
#######################################################################
WORKDIR  /
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update 
RUN apt-get install -y libopencv-dev \
    ros-noetic-map-server \
    ros-noetic-amcl \
    ros-noetic-teleop-twist-joy \
    ros-noetic-camera-calibration \
    ros-noetic-gmapping \
    ros-noetic-ros-numpy \
    ros-noetic-robot-state-publisher

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]


#######################################################################
##                        install realsense                          ##
#######################################################################

RUN sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main" -u
RUN sudo apt-get -y install librealsense2-dkms
RUN sudo apt-get -y install librealsense2-utils
RUN sudo apt-get -y install librealsense2-dev
RUN sudo apt-get -y install librealsense2-dbg

RUN apt-get install -y ros-$ROS_DISTRO-realsense2-camera
RUN apt-get install -y ros-$ROS_DISTRO-realsense2-description

RUN pip install pyrealsense2

#######################################################################
##                        install yolov5                             ##
#######################################################################
RUN pip install yolov5
RUN pip install torch
# RUN pip install open3d
# RUN pip install openvino

# #######################################################################
# ##                        install openvino                           ##
# #######################################################################
# RUN git clone -b 2022.1.0 https://github.com/openvinotoolkit/openvino.git && \
#     cd openvino && \
#     git submodule update --init --recursive && \
#     chmod +x install_build_dependencies.sh && \
#     bash install_build_dependencies.sh && \
#     mkdir build && cd build && \
#     cmake -DCMAKE_BUILD_TYPE=Release .. && \
#     make --jobs=$(nproc --all) && \
#     cd /

#######################################################################
##                            delete cash                            ##
#######################################################################
RUN rm -rf /var/lib/apt/lists/*


#######################################################################
##                         make sudoer user                          ##
#######################################################################

ARG USER
ARG USER_ID
 
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw" >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

#######################################################################
##              make user accessible to usb and so on                ##
#######################################################################
RUN adduser ${USER} dialout
RUN adduser ${USER} tty
RUN adduser ${USER} video
RUN adduser ${USER} root

# USER root
USER ${USER}
RUN echo "export PS1='\[\e[1;31;40m\]OBJECT_DETECTOR\[\e[0m\] \u:\w\$ '">> ~/.bashrc
RUN echo "source /ros_setting.sh">> ~/.bashrc

ARG workspace
WORKDIR ${workspace}