cmake_minimum_required(VERSION 3.25)
project(corridor_planning)

find_package(aduulm_cmake_tools REQUIRED)
aduulm_init_ros2_package()

install(PROGRAMS
  scripts/path_modifications.py
  scripts/sim_params.py
  scripts/sim_vehicle.py
  scripts/simulation.py
  DESTINATION lib/${PROJECT_NAME}
)

# Finish the package creation and generate install() commands
aduulm_create_ros2_package()