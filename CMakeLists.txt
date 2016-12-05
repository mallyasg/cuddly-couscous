# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(RandomForest)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

set (SOURCE_DIR ./source)

aux_source_directory(${SOURCE_DIR} SOURCES)
set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -g")
message(STATUS ${SOURCES})

# Declare the executable target built from your sources
add_executable(RandomForest ${SOURCES})

# Link your application with OpenCV libraries
target_link_libraries(RandomForest ${OpenCV_LIBS})

# Link your application with Eigen3 libraries
target_link_libraries(ConvexHull ${EIGEN3_LIBS})
