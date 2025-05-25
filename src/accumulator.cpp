#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_config.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h> 

#include <tf2/convert.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <string>

class Accumulator {
public:
    Accumulator() {
        cloud_sub_ = nh_.subscribe("/clustered_cloud", 1, &Accumulator::cloudCallback, this);
        odom_sub_ = nh_.subscribe("/fast_limo/state", 1, &Accumulator::odomCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_cloud", 1);

        ros::NodeHandle nh_p("~");
        nh_p.param<std::string>("save_path", save_path, "");

        ROS_WARN("TREESCOPE::ACCUMULATOR saving accumulated pointcloud in %s", save_path.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input, *cloud);

        // Transform pointcloud to global frame
        pcl::transformPointCloud (*cloud, *cloud, transform);

        // Publish clustered cloud
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = input->header;
        output.header.frame_id = global_frame;
        cloud_pub_.publish(output);

    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
        tf2::fromMsg(msg->pose.pose, transform);
        global_frame = msg->header.frame_id;
    }

    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_, odom_sub_;
    ros::Publisher cloud_pub_;

    std::string save_path;
    std::string global_frame;

    Eigen::Affine3d transform;

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "accumulator");
    Accumulator accumulator;
    
    // Start spinning (async)
    ros::AsyncSpinner spinner(0);
    spinner.start();

    ros::waitForShutdown();
    return 0;
}