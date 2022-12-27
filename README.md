# yolov5

## Software setup
1. Clone
    ```shell
    $ git clone git@github.com:Takahashi-Lab-Keio/ytlab_yolov5.git
    ```
2. Build docker image
    ```shell
    $ cd ytlab_yolov5/docker
    $ ./build.sh
    ```
3. Initial launch docker container
    ```shell
    $ cd ytlab_yolov5/docker
    $ ./run.sh
    DESTINATION$ cm
    DESTINATION$ exit
    ```

## Launch
```shell
roslaunch whill_path kazu_human_detection.launch 
```
# ouster_yolov5


roslaunch whill_path map_viewer.launch 