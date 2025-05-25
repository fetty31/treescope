cmake_minimum_required(VERSION 3.0.2)
project(treescope)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  tf2_ros
  tf2
  image_transport
  cv_bridge
  sensor_msgs
  std_msgs
  visualization_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL REQUIRED)

add_message_files(
  FILES
  Cluster.msg
  ClusterArray.msg
  YoloDetect2D.msg
  YoloDetect2DMask.msg
  YoloDetect2DArray.msg
  YoloDetect2DMaskArray.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
    visualization_msgs
    pcl_ros
    tf2
    tf2_ros
    image_transport
    cv_bridge
    pcl_conversions
    message_runtime
  DEPENDS EIGEN3 PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(clustering src/clustering.cpp) 
target_link_libraries(clustering ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(clustering ${catkin_EXPORTED_TARGETS})

add_executable(adaptive_clustering src/adaptive_clustering.cpp) 
target_link_libraries(adaptive_clustering ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(adaptive_clustering ${catkin_EXPORTED_TARGETS})

add_executable(tracking 
              src/tracking.cpp
              src/kalman.cpp
              src/hungarian.cpp
              ) 
target_link_libraries(tracking ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(tracking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(cropfilter src/cropfilter.cpp) 
target_link_libraries(cropfilter ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(accumulator src/accumulator.cpp) 
target_link_libraries(accumulator ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(image_dtype src/image_process/dtype.cpp) 
target_link_libraries(image_dtype ${catkin_LIBRARIES})