#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <treescope/ClusterArray.h>
#include <treescope/Cluster.h>

#include <limits>
#include <chrono>

class AdaptiveClustering {
public:
    AdaptiveClustering() {
        sub_ = nh_.subscribe(nh_.getNamespace() + "/no_ground_cloud", 1, &AdaptiveClustering::cloudCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(nh_.getNamespace() + "/clustered_cloud", 1);
        noncloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(nh_.getNamespace() + "/nonclustered_cloud", 1);
        centroid_pub_ = nh_.advertise<geometry_msgs::PointStamped>(nh_.getNamespace() + "/cluster_centroids", 1);
        cluster_pub_  = nh_.advertise<treescope::ClusterArray>(nh_.getNamespace() + "/clusters", 1);

        ros::NodeHandle nh_p("~");
        nh_p.param<double>("Tolerance", cluster_tolerance, 0.2);
        nh_p.param<int>("MaxNumPoints", max_cluster_size, 25000);
        nh_p.param<int>("MinNumPoints", min_cluster_size, 20);

        nh_p.param<double>("WIDTH_THRESHOLD", WIDTH_THRESHOLD, 3.0);
        nh_p.param<double>("HEIGHT_THRESHOLD", HEIGHT_THRESHOLD, 3.0);
        nh_p.param<double>("LENGTH_THRESHOLD", LENGTH_THRESHOLD, 3.0);
        nh_p.param<double>("HEIGHT_MIN", HEIGHT_MIN, 3.0);
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input) {

        const auto start = std::chrono::system_clock::now();

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input, *cloud);

        std::vector<pcl::PointIndices> all_clusters;
        std::vector<std::pair<float, float>> bands = {
            {0.0, 10.0},
            {10.0, 20.0},
            {20.0, 40.0}
        };

        for (auto& band : bands) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr band_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            std::vector<int> band_indices;

            for (size_t i = 0; i < cloud->points.size(); ++i) {
                float dist = std::sqrt(cloud->points[i].x * cloud->points[i].x +
                                       cloud->points[i].y * cloud->points[i].y);
                if (dist >= band.first && dist < band.second) {
                    band_cloud->points.push_back(cloud->points[i]);
                    band_indices.push_back(i);
                }
            }

            if (band_cloud->points.empty()) continue;

            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(band_cloud);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(this->cluster_tolerance + 0.01 * band.first);
            ec.setMinClusterSize(this->min_cluster_size);
            ec.setMaxClusterSize(this->max_cluster_size);
            ec.setSearchMethod(tree);
            ec.setInputCloud(band_cloud);
            ec.extract(cluster_indices);

            for (auto& cluster : cluster_indices) {
                for (auto& idx : cluster.indices) {
                    idx = band_indices[idx];
                }
                all_clusters.push_back(cluster);
            }
        }

        treescope::ClusterArray cluster_array_msg;
        cluster_array_msg.header = input->header;

        // Create colored cloud for visualization
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr nonclustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        clustered_cloud->header = cloud->header;
        nonclustered_cloud->header = cloud->header;
        for (const auto& indices : all_clusters) {

            uint8_t r = rand() % 256, g = rand() % 256, b = rand() % 256;

            float x_min = std::numeric_limits<float>::max(), x_max = -x_min;
            float y_min = x_min, y_max = -x_min;
            float z_min = x_min, z_max = -x_min;

            Eigen::Vector3f accumulator {0, 0, 0};

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& idx : indices.indices) {
                pcl::PointXYZRGB point;
                point.x = cloud->points[idx].x;
                point.y = cloud->points[idx].y;
                point.z = cloud->points[idx].z;
                point.r = r;
                point.g = g;
                point.b = b;
                accumulator[0] += point.x;
                accumulator[1] += point.y;
                accumulator[2] += point.z;
                x_min = std::min(x_min, point.x);
                x_max = std::max(x_max, point.x);
                y_min = std::min(y_min, point.y);
                y_max = std::max(y_max, point.y);
                z_min = std::min(z_min, point.z);
                z_max = std::max(z_max, point.z);
                cluster->push_back(point);
            }

            float width = x_max - x_min;
            float length = y_max - y_min;
            float height = z_max - z_min;

            // Compute and publish centroid
            Eigen::Vector3f centroid;
            computeLowCentroid(cluster, 0.5f, centroid);

            geometry_msgs::PointStamped centroid_msg;
            centroid_msg.header = input->header;
            centroid_msg.point.x = centroid[0];
            centroid_msg.point.y = centroid[1];
            centroid_msg.point.z = centroid[2];
            centroid_pub_.publish(centroid_msg);
            
            // Filter cluster by volume and publish
            treescope::Cluster cluster_msg;
            uint64_t id = 0;

            if (keepVolume(width, length, height)) {
                *clustered_cloud += *cluster;

                sensor_msgs::PointCloud2 pcl_msg;
                pcl::toROSMsg(*cluster, pcl_msg);
                pcl_msg.header = input->header;

                cluster_msg.id = id++;
                cluster_msg.cloud = pcl_msg;
                cluster_msg.centroid.x = static_cast<double>(centroid[0]); 
                cluster_msg.centroid.y = static_cast<double>(centroid[1]); 
                cluster_msg.centroid.z = static_cast<double>(centroid[2]); 
                cluster_array_msg.clusters.push_back(cluster_msg);
            } else {
                *nonclustered_cloud += *cluster;
            }

        }

        // Publish clusters
        cluster_pub_.publish(cluster_array_msg);

        // Publish clustered cloud
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*clustered_cloud, output);
        output.header = input->header;
        cloud_pub_.publish(output);

        // Publish nonclustered cloud
        sensor_msgs::PointCloud2 output_non;
        pcl::toROSMsg(*nonclustered_cloud, output_non);
        output_non.header = input->header;
        noncloud_pub_.publish(output_non);

        const auto end = std::chrono::system_clock::now();

        std::chrono::duration<float> elapsed_time = end - start;

        ROS_INFO("Adaptive clustering took %f ms", elapsed_time.count()*1000.0f);

    }

    // Whether to keep the cluster based on its volume
    bool keepVolume(float width, float length, float height) {
        if( (width < WIDTH_THRESHOLD) && (height < HEIGHT_THRESHOLD) && (length < LENGTH_THRESHOLD) && (height > HEIGHT_MIN) )
            return true;
        else
            return false;
    }

    void computeLowCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cluster, float min_z, Eigen::Vector3f& centroid){
        Eigen::Vector3f accumulator {0, 0, 0};
        int c = 0;
        for(int i=0; i < cluster->points.size(); i++){
            auto& point = cluster->points[i];
            if(point.z <= min_z){
                accumulator[0] += point.x;
                accumulator[1] += point.y;
                accumulator[2] += point.z;
                c++;
            }
        }
        centroid = accumulator;
        centroid /= (c > 0) ? static_cast<float>(c) : 1.0f;
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher cloud_pub_, centroid_pub_, noncloud_pub_, cluster_pub_;

    int max_cluster_size, min_cluster_size;
    double cluster_tolerance;

    double WIDTH_THRESHOLD;
    double HEIGHT_THRESHOLD;
    double HEIGHT_MIN; 
    double LENGTH_THRESHOLD;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "AdaptiveClustering");
    AdaptiveClustering AdaptiveClustering;
    ros::spin();
    return 0;
}