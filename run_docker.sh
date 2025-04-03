#!/bin/bash
tput setaf 1
if [ ! -z "${DOCKER_MACHINE_NAME}" ]; then
  >&2 echo "Error: You probably are already inside a docker container!"
  tput sgr 0
  exit 1
elif [ ! -e /var/run/docker.sock ]; then
  >&2 echo "Error: Either docker is not installed or you are already inside a docker container!"
  tput sgr 0
  exit 1
fi
tput sgr 0

MAC_ADDRESS=$(ip link | grep -m1 ether | cut -d " " -f6)
IP_ADDRESS=$(ip -4 addr show scope global | grep -m1 inet | awk "{print \$2}" | cut -d / -f 1)

SBX_DIR=$(realpath $(dirname $0))

# default values for image name and tag
IMAGE_NAME_SUFFIX="dynamic_objective_mpc"
ROS_DISTRO="jazzy"
TAG=$ROS_DISTRO

WORKSPACE="${SBX_DIR}"

# read image name, tag and container name from arguments if given
while [[ $# -gt 0 ]]; do
  key="$1"
  if [ -z ${EXEC_ARGS+x} ]; then
    case ${key} in
      -t|--tag)
      TAG="$2"
      shift # past argument
      shift # past value
      ;;
      -s|--stop)
      SKIP_PULL=true
      STOP_CONTAINER=true
      shift # past argument
      ;;
      -sa|--stop-all)
      STOP_ALL_CONTAINERS=true
      shift # past argument
      ;;
      -rma|--rm-all)
      REMOVE_ALL_CONTAINERS=true
      shift # past argument
      ;;
      --skip-pull)
      SKIP_PULL=true
      shift # past argument
      ;;
      --service)
      AS_SERVICE=true
      shift # past argument
      ;;
      -w|--workspace)
      WORKSPACE="$2"
      shift # past argument
      shift # past value
      ;;
      --direct)
      DIRECT=true
      shift # past argument
      ;;
      -h|--help)
      SHOW_HELP=true
      break
      ;;
      --exec)
      EXEC_ARGS=("")
      shift # past argument
      ;;
      *)  # unknown options are passed to docker
      CMD_EXTRA_ARGS+=("$1")
      shift # past argument
      ;;
    esac
  else
    EXEC_ARGS+=("$1") # save it in an array for later
    shift
  fi
done

# create full name
IMAGE_NAME="${IMAGE_NAME_SUFFIX}:${TAG}"

# use shell in case no exec args are given
if [ ${#EXEC_ARGS[@]} -eq 0 ]; then
  EXEC_ARGS=(${SHELL})
fi

# help
if [[ ${SHOW_HELP} = true ]]; then
  echo "Usage: ./run_docker.bash [Options] [Arguments]"
  echo ""
  echo "Options:"
  echo " * -t  / --tag TAG:      Docker image tag (default from .gitlab-ci.yaml)"
  echo " * -s  / --stop:         Stop this container"
  echo " * -sa / --stop-all:     Stop all running containers that were started with this script (ignoring containers with custom names)"
  echo " * -rma / --rm-all:      Stop and remove all existing containers"
  echo " * --skip-pull:          Skip pulling docker image from registry"
  echo " * --service:            Run container as service (non-interactive, not started and as root)"
  echo " * -w  / --workspace WS: Docker workspace (default: sandbox directory)"
  echo " * --direct:             Start container directly and without entrypoint instead of in the background and connecting afterwards"
  echo " * -h  / --help:         Show this message"
  echo " * --exec ARG [ARG ..]:  All subsequent arguments are executed within the Docker container"
  echo ""
  echo "All unknown arguments are passed to docker."
  exit 0
fi

# stop containers
if [[ ${STOP_ALL_CONTAINERS} = true ]]; then
  running_containers=$(docker ps | grep '\-docker-runner-' | awk '{ print $1 }')
  if [[ ! -z ${running_containers} ]]; then
    docker stop ${running_containers[@]}
  fi
  echo "Stopped all containers created by this script (ignoring containers with custom names)."
  exit 0
fi

# remove containers
if [[ ${REMOVE_ALL_CONTAINERS} = true ]]; then
  running_containers=$(docker ps -q)
  if [[ ! -z ${running_containers} ]]; then
    docker kill ${running_containers}
  fi
  stopped_containers=$(docker ps -aq)
  if [[ ! -z ${stopped_containers} ]]; then
    docker rm ${stopped_containers}
  fi
  echo "Killed and removed all existing containers."
  exit 0
fi

# for the lazy and reckless: allow access to X session
# this is apparently unsafe, see: http://wiki.ros.org/docker/Tutorials/GUI
xhost +local:root >/dev/null 2>/dev/null

# Add docker to relevant user groups
grpnames=""
for res in $(LC_ALL=en_US.UTF-8 id | python3 -c 'import re,sys; msg=sys.stdin.read().split("groups=")[1]; matches = re.findall(r"(\d+)\((\w+)\)", msg); [print("{};{}".format(gid, grpn)) for gid, grpn in matches]'); do
  data=(${res//;/ });
  # echo ${data[0]}, ${data[1]};
  grpnames+=" --group-add ${data[0]}"
done

# Run the following line on the host machine to enable core dumps in docker (commented out due to sudo requirement):
#echo '/tmp/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern

docker_ver=`docker --version | sed -e 's/.*version \([^\.]\+\)\..*/\1/'`

lspci | grep NVIDIA >/dev/null 2>&1
no_gpu=$?
which nvidia-container-toolkit >/dev/null 2>&1
no_nvidia_docker=$?

if [[ ${AS_SERVICE} = true ]]; then
  DOCKER_CMD="create"
  INTERNAL_ARGS=(
    --user 0:0
    -t
  )

else
  DOCKER_CMD="run"
  INTERNAL_ARGS=(
    -v "${HOME}":"${HOME}"
    -v "${HOME}/.local/lib/python3.10"
    --user $(id -u):$(id -g)
    --rm  # remove container after stopping
  )
fi

if [ ${no_gpu} -ne 0 ] || [ ${no_nvidia_docker} -ne 0 ]; then
  tput setaf 2
  if [[ ${no_gpu} -eq 0 && ${no_nvidia_docker} -ne 0 ]]; then
    echo "Your PC seems to have an NVIDIA GPU, but nvidia-docker could not be found. GPU support will be disabled inside docker."
  else
    echo "No GPU found. GPU support will be disabled inside docker."
  fi
  tput sgr 0
  DOCKER="docker ${DOCKER_CMD}"
elif [ ${docker_ver} -lt 19 ]; then
  DOCKER="nvidia-docker ${DOCKER_CMD}"
else
  DOCKER="docker ${DOCKER_CMD} --gpus all"
fi

HOSTS=`cat /etc/hosts | grep add_to_docker | sed -e 's/\s*#.*$//' -e 's/\(.*\)\s\+\(.*\)/--add-host=\2:\1/'`

EXTRA_ARGS=""
function get_extra_args {
  EXTRA_ARGS_FILE=""
  if [ -f "$1" ]; then
    while read -r line || [ "${line}" ]; do
      EXTRA_ARGS_FILE+=" ${line}"
    done < <(sed '/^[[:blank:]]*#/d;s/#.*//' "$1")
    EXTRA_ARGS+=" "
    EXTRA_ARGS+=`eval "echo \"${EXTRA_ARGS_FILE}\""`
    return 0
  fi
  return 1
}

get_extra_args "${SBX_DIR}/.extra_docker_args" || \
get_extra_args "${HOME}/.extra_docker_args" || \
get_extra_args "/home/aduulm/.extra_docker_args" || \
#echo "${EXTRA_ARGS}"

dsp="$(echo "${DISPLAY}" | sed 's/\.0$//')"

if [[ ! ${SKIP_PULL} = true ]]; then
  HOST_REACHABLE=1;
  timeout 0.5 nc -zv mrm-git.e-technik.uni-ulm.de 4567 >/dev/null 2>&1 || HOST_REACHABLE=0;

  if [ ${HOST_REACHABLE} -eq 1 ]; then
    echo "Pulling image..."
    docker pull ${IMAGE_NAME}
  else
    echo "No connection to host so no pulling possible."
  fi
fi

DOCKER_ARGS=(
  -v "${SBX_DIR}":"${SBX_DIR}"
  -v ${WORKSPACE}:${WORKSPACE}
  -e COLCON_HOME=${SBX_DIR} \
  -v /tmp:/tmp:rw
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v /etc/group:/etc/group:ro
  -v /etc/passwd:/etc/passwd:ro
  -v /etc/shadow:/etc/shadow:ro
  -e DISPLAY="${dsp}"
  -e DOCKER_MACHINE_NAME="${IMAGE_NAME_SUFFIX}:${TAG}"
  --network=host
  --ulimit core=99999999999:99999999999
  --ulimit nofile=1024 # makes forking processes faster, see https://github.com/docker/for-linux/issues/502
  --privileged
  --add-host=localhost:${IP_ADDRESS}
  --add-host=$(cat /etc/hostname):${IP_ADDRESS}
  ${HOSTS}
  --group-add sudo
  --group-add dialout
  ${grpnames}
  --ipc=host
  --device /dev/dri
  -w ${WORKSPACE}
  ${INTERNAL_ARGS[@]}
  ${EXTRA_ARGS}
  ${CMD_EXTRA_ARGS[@]}
)

# create container name if not given
if [[ -z ${CONTAINER_NAME+x} ]] && [[ ! ${DIRECT} = true ]]; then
  ARGS_MD5=$(echo "${DOCKER_ARGS[@]}" | md5sum | awk '{ print $1 }')
  IMAGE_MD5=`docker images --no-trunc --quiet $IMAGE_NAME | sed -e 's/^sha256://'`
  CONTAINER_NAME=$(echo ${SBX_DIR} | sed -e 's+/+_+g' -e 's/^_*//')-docker-runner-${IMAGE_MD5}-${ARGS_MD5}
fi

# stop container
if [[ ${STOP_CONTAINER} = true ]]; then
  if [ -z ${CONTAINER_NAME+x} ]; then
    echo "Cannot stop direct container without given name"
    exit 1
  else
    docker stop "${CONTAINER_NAME}"
    exit $?
  fi
fi

# set container name
if [ ! -z ${CONTAINER_NAME+x} ]; then
  DOCKER_ARGS+=(--name "${CONTAINER_NAME}" -e CONTAINER_NAME="${CONTAINER_NAME}")
fi

# set image name
DOCKER_ARGS+=(${IMAGE_NAME})


if [[ ${AS_SERVICE} = true ]]; then
  echo "Creating a new docker container"
  ${DOCKER} ${DOCKER_ARGS[@]} ${EXEC_ARGS[@]}

  # check return code
  ret=$?
  if [ ${ret} -ne 0 ]; then
      tput setaf 1
      echo "Failed to create the container using the following command:"
      echo "${DOCKER} ${DOCKER_ARGS[@]} ${EXEC_ARGS[@]}"
      tput sgr 0
  fi
  exit ${ret}

elif [[ ${DIRECT} = true ]]; then
  echo "Running a new docker container"
  ${DOCKER} -it ${DOCKER_ARGS[@]} ${EXEC_ARGS[@]}
  exit $?

else
  # set entrypoint and detach
  DOCKER_ARGS=(-d --entrypoint "${SBX_DIR}/docker/content/daemon-entrypoint.sh" "${DOCKER_ARGS[@]}")

  docker ps | tail -n +2 | awk '{print $(NF)}' | grep -e "^${CONTAINER_NAME}\$" >/dev/null 2>&1
  CONTAINER_EXISTS=$?

  if [ ${CONTAINER_EXISTS} -eq 0  ]; then
    echo "Connecting to existing container."
  else
    echo "Creating a new docker container"
    ${DOCKER} ${DOCKER_ARGS[@]}

    # check return code
    ret=$?
    if [ ${ret} -ne 0 ]; then
      tput setaf 1
      echo "Failed to create the container using the following command:"
      echo "${DOCKER} ${DOCKER_ARGS[@]}"
      tput sgr 0
      exit ${ret}
    fi

    echo "Connecting to container."
  fi

  NUM_CONTAINERS=`docker ps | grep '\-docker-runner-' | wc -l`
  tput setaf 2
  if [ ${NUM_CONTAINERS} -gt 1 ]; then
    echo "Just so you know, there are currently ${NUM_CONTAINERS} sandbox containers running. You can use 'docker ps' to check for containers that are no longer needed, stop this container with './run_docker.sh -s' or stop all containers with './run_docker.sh -sa'."
  fi
  echo
  echo "If graphical applications fail to run inside docker, check if the DISPLAY variable is set correctly (check the value inside and outside of docker)."
  tput sgr 0

  docker exec -it "${CONTAINER_NAME}" /usr/bin/docker-entrypoint.sh ${EXEC_ARGS[@]}
  exit $?
fi
