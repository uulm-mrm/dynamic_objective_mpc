#!/bin/zsh
set -e

export GPG_TTY=$(tty)

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

source /opt/ros/jazzy/setup.zsh
source /opt/aduulm/setup.zsh

echo "Welcome to the docker of the dynamic_objective_mpc. You can run and modify the code from here."
echo "Run the following commands"
echo "colcon build"
echo "source colcon_build/install/setup.zsh"
echo "ros2 run corridor_planning simulation.py --ros-args -p baseline:=-1 -p track_switch:=6"

exec "$@"
