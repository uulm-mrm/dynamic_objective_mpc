# 


## Description
This repository contains the code of the IV2025 paper called "Dynamic Objective MPC for Motion Planning of Seamless 
Docking Maneuvers"

>O. Schumann, M. Buchholz and K. Dietmayer, "Dynamic Objective MPC for Motion Planning of Seamless Docking Maneuvers," 2025 IEEE Intelligent Vehicles Symposium (IV), Cluj-Napoca, Romania, 2025, pp. 132-139, doi: 10.1109/IV64158.2025.11097771.
> 
This algorithm is based on a model predictive contouring controller (MPCC) created in acados, which is improved by the 
methods proposed in this paper. 
These changes allow for high-precision motion planning along a reference path while being able to reach specific goal 
poses at the same time without stopping or switching to another controller.
Especially in docking scenarios, this is a common use case.

![GUI](img/gui.png)
## Paper
[arXiv](http://arxiv.org/abs/2504.03280)

[IEEE-Explore](https://doi.org/10.1109/IV64158.2025.11097771)

[Oparu](https://oparu.uni-ulm.de/items/956a606e-3145-488b-b1a3-85bd32b1b410)

## Citation
```
@INPROCEEDINGS{11097771,
  author={Schumann, Oliver and Buchholz, Michael and Dietmayer, Klaus},
  booktitle={2025 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Dynamic Objective MPC for Motion Planning of Seamless Docking Maneuvers}, 
  year={2025},
  volume={},
  number={},
  pages={132-139},
  doi={10.1109/IV64158.2025.11097771}}

```

## Videos
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/28X5zaHW6bs/0.jpg)](https://www.youtube.com/watch?v=28X5zaHW6bs)

## Setup
1. Build the docker image
```shell
cd docker && ./build.sh && cd ..
```
2. Run the docker
```shell
./run_docker.sh
```
3. Build and source the module
```shell
colcon build
source colcon_build/install/setup.zsh
```
4. Run
```shell
ros2 run corridor_planning simulation.py --ros-args -p baseline:=-1 -p track_switch:=6
```

Different configurations of the planning algorithm can be run, depending on some parameters:

**baseline** \
-1: dynamic objective MPC \
0: separated motion plans \
1: switched MPCs 


**track_switch (tracks used in this paper are bold)** \
0: racing track \
1: track with switch and goal pose \
2: partially narrow track with switch and goal pose \
3: left right switches \
4: huge left turn \
5: narrow gap \
6: track with switch and goal
track with switch and goal pose
   
