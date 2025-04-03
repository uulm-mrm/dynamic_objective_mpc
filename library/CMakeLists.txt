cmake_minimum_required(VERSION 3.10)
project(corridor_planning_lib)

# Get aduulm cmake macros for coloring, package_version extraction and find_aduulm_package()
find_package(aduulm_cmake_tools REQUIRED)
get_version_string_from_package_xml(.)
show_build_info("library")
#setup_git_hooks()

find_aduulm_package(aduulm_logger_lib REQUIRED)

# install python code
# PYTHON_INSTALL_DIR is defined by aduulm_cmake_tools
install(DIRECTORY ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/
        DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

set(INCLUDE_INSTALL_DIR include/ CACHE STRING "Path to installed include directory")
set(LIB_INSTALL_DIR lib/ CACHE STRING "Path to installed lib directory")
set(BIN_INSTALL_DIR bin/ CACHE STRING "Path to installed bin directory")

# INSTALLATION (for libraries)
set(PACKAGE_LIBRARY_VERSION ${package_version})

# Create package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PACKAGE_LIBRARY_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

# Create package config file (to enable find_package())
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/config.cmake.in" # this file defines the include dirs and libs and may need to be adapted for your package/library
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR
)

# install CMake config and version file
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME})

# Build optimizers
# TODO suppress numpy warnings
install(CODE "execute_process(COMMAND python3 install.py ${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/
WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/mpc/)")

