#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>

#include <vector>

class CropFilter {
public:
    CropFilter() {
        sub_ = nh_.subscribe("/fast_limo/pointcloud_local", 1, &CropFilter::cloudCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);

        ros::NodeHandle nh_p("~");
        nh_p.param<std::vector<float>>("CropFilter/Min", crop_box_min,  {-1.0, -1.0, -1.0} );
        nh_p.param<std::vector<float>>("CropFilter/Max", crop_box_max,  {1.0, 1.0, 1.0} );

        this->crop_filter.setNegative(true);
        this->crop_filter.setMin(Eigen::Vector4f(crop_box_min[0], crop_box_min[1], crop_box_min[2], 1.0));
        this->crop_filter.setMax(Eigen::Vector4f(crop_box_max[0], crop_box_max[1], crop_box_max[2], 1.0));
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input, *cloud);

        this->crop_filter.setInputCloud(cloud);
        this->crop_filter.filter(*cloud);

        // Publish clustered cloud
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = input->header;
        cloud_pub_.publish(output);

    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher cloud_pub_;

    pcl::CropBox<pcl::PointXYZ> crop_filter;
    std::vector<float> crop_box_max, crop_box_min;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cropfilter");
    CropFilter cropfilter;
    ros::spin();
    return 0;
}