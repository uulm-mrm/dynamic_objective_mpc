# Base ubuntu
FROM osrf/ros:jazzy-desktop-full

USER $USER

# Suppress interactions during build
ENV DEBIAN_FRONTEND="noninteractive"
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# set locale
RUN apt update && apt install locales -y
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

# First ensure that the Ubuntu Universe repository is enabled.
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository universe

# Install development tools
RUN apt update && apt install -y \
    tmux \
    nano \
    vim \
    zsh \
    wget \
    build-essential \
    cmake \
    git \
    python3-pip \
    clangd \
    htop \
    gitg \
    git-lfs \
    lsb-release \
    clang-format \
    libglfw3-dev \
    libglew-dev

# install casadi and acados
RUN pip3 install --no-cache-dir --break-system-packages casadi==3.7.0 && \
    cd /opt && \
    git clone --branch v0.4.4 --recurse-submodules https://github.com/acados/acados.git && \
    mkdir -p acados/build && \
    cd acados/build && \
    cmake .. -DACADOS_PYTHON=ON -DACADOS_SILENT=ON && \
    make install -j$(nproc --all) && \
    cd ../bin && \
    wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux && \
    mv t_renderer-v0.0.34-linux t_renderer && \
    chmod +x t_renderer
ENV ACADOS_SOURCE_DIR="/opt/acados"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/opt/acados/lib"
RUN pip3 install --break-system-packages -e $ACADOS_SOURCE_DIR/interfaces/acados_template

# install imviz, the pip version is currently not buildable on 24.04
RUN cd /opt && \
    git clone --branch develop https://github.com/joruof/imviz.git && \
    cd imviz && python3 -m pip install . --break-system-packages && \
    cd ../ && \
    rm -r imviz

# install other python packages
RUN pip3 install --break-system-packages numba termcolor pandas

# clone needed modules from MRM: Attention the docker must be rebuild if you need a newer version of them
ENV ROS2_SOURCE_DIR="/tmp/ros2_jazzy_sources"
ENV ROS2_INSTALL_DIR="/opt/aduulm"
SHELL ["/bin/zsh", "-c"]
RUN mkdir -p ${ROS2_SOURCE_DIR}/src && \
    source /opt/ros/jazzy/setup.zsh && \
    cd ${ROS2_SOURCE_DIR} && \
    git clone --branch develop --recurse-submodules https://github.com/uulm-mrm/aduulm_cmake_tools.git && \
    cd aduulm_cmake_tools && git submodule update --init --recursive && cd .. && \
    git clone --branch develop https://github.com/uulm-mrm/aduulm_logger.git && \
    colcon build \
          --install-base ${ROS2_INSTALL_DIR} \
          --cmake-args \
            " --no-warn-unused-cli" \
            " -DCMAKE_BUILD_TYPE=Release" \
    && \
    rm -r ${ROS2_SOURCE_DIR}

# Add graphics capability to allow rviz to run with GPU acceleration
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# set entrypoint
COPY ./docker/content/docker-entrypoint-ros2.sh /usr/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]


