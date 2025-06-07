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

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>

#include <treescope/ClusterArray.h>

#include <std_srvs/Trigger.h>

#include <tf2/convert.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <string>
#include <boost/make_shared.hpp>

class Accumulator {
public:
    Accumulator() {
        cloud_sub_ = nh_.subscribe("/clustered_cloud", 1, &Accumulator::cloudCallback, this);
        odom_sub_  = nh_.subscribe("/fast_limo/state", 1, &Accumulator::odomCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_cloud", 1);

        service_       = nh_.advertiseService("/accumulator/clear_clusters", &Accumulator::clearMap, this);
        clear_service_ = nh_.advertiseService("/accumulator/save_clusters", &Accumulator::saveMap, this);

        ros::NodeHandle nh_p("~");
        nh_p.param<std::string>("save_path", save_path, "");

        map = pcl::PointCloud<pcl::PointXYZ>::Ptr (boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>());

        voxel_filter.setLeafSize(0.5, 0.5, 0.5);

        ROS_WARN("TREESCOPE::ACCUMULATOR saving accumulated pointcloud in %s", save_path.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input) {

        mtx_.lock();

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

        voxel_filter.setInputCloud(cloud);
        voxel_filter.filter(*cloud);

        *map += *cloud;

        mtx_.unlock();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
        tf2::fromMsg(msg->pose.pose, transform);
        global_frame = msg->header.frame_id;
    }

    bool clearMap(std_srvs::Trigger::Request &req,
                    std_srvs::Trigger::Response& res)
    {
        mtx_.lock();

        map->clear();   
        res.success = true;
        res.message = "Map cleared succesfully";

        mtx_.unlock();
        return true;
    }

    bool saveMap(std_srvs::Trigger::Request &req,
                    std_srvs::Trigger::Response& res)
    {
        mtx_.lock();

        ROS_INFO("Treescope::Accumulator:: Saving clusters to %s", save_path.c_str());

        res.message = "";
        res.success = true;
        res.message = "All ok";

        std::cout << "Before Clustering\n";

        // Euclidean clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(map);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.2); // 20cm
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(map);
        ec.extract(cluster_indices);

        std::cout << "Clustering done\n";
        std::cout << "  number of clusters: " << cluster_indices.size() << std::endl;

        int id = -1;
        for (const auto& indices : cluster_indices) {

            std::cout << "cluster " << id << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : indices.indices)
                cluster->push_back(map->points[idx]);

            std::cout << "cluster pointcloud filled\n";

            std::string file_name = "cluster_" + std::to_string(++id) + ".pcd";
            if(pcl::io::savePCDFileBinary(save_path+file_name, *cluster)!=0){
                ROS_ERROR("Treescope::Accumulator:: Failed to save pcd to %s", save_path.c_str());
                res.success = false;
                res.message += "Something failed while saving pcd files: " + file_name + " ";
            }
        }

        mtx_.unlock();

        return true;
    }


    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_, odom_sub_;
    ros::Publisher cloud_pub_;

    ros::ServiceServer service_, clear_service_;

    std::string save_path;
    std::string global_frame;

    Eigen::Affine3d transform;

    pcl::PointCloud<pcl::PointXYZ>::Ptr map;

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;

    std::mutex mtx_;

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