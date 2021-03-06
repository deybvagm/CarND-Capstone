This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Project architecture
This project is intended to run only on the simulator and was run using docker container

![image](imgs/architecture.png)

#### Perception module
This module is responsible for sensing the environment and communicate if an action should be performed. Specifically for this project, an action should be taken if there is a red traffic light in front of the car. To do this I have used two models, one for detecting the traffic light and one for classifying the traffic light color into red, yellow, green or unknown:

- **Traffic light detection**: For this task, I used the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) with a pre-trained model using the COCO dataset. This model is able to recognize many objects, including traffic lights which is is represented with class 10 from the model output. Within the docker container, the model can be accessed with the path `models/detection/frozen_inference_graph.pb` in the `tl_detector` module

- **Traffic light color classification**: For the classification task, a CNN was implemented using Keras. The CNN architecture was the LeNet5 which is presented below:

![LeNet5 model](imgs/model.png)

Tu run the model inside the docker container, the model should be located in the path `models/classification/classification_model.h5` inside the `tl_detector` module

### Planning module
This module(`waypoint_updater`) subscribes to the `/traffic_waypoint` message that contains the closest waypoint where there is a red traffic light in order to change the car's speed gradually until it stops at the waypoint

### Control module
This module (`twist_controller`) is responsible for sending to the car commands related to the brake, throttle and steering values. It takes information from other modules about the linear and angular velocity and also the current velocity of the car. To give the values for steering, throttle and brake
a PID controller is implemented in `pid.py` as well as a yaw controller in `yaw_controller.py`
