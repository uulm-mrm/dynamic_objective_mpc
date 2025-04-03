#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROS_DISTRO="jazzy"
DOCKER_IMAGE_NAME="dynamic_objective_mpc"

docker build \
    -t $DOCKER_IMAGE_NAME:$ROS_DISTRO \
    -f Dockerfile ${SCRIPT_DIR}/.. \
    "$@"

