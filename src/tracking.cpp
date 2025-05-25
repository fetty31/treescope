#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <treescope/kalman.h>
#include <treescope/hungarian.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <treescope/ClusterArray.h>

#include <string>
#include <vector>
#include <map>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

static const int COOLDOWN_FRAMES = 2;

// Tracked object structure
struct TrackedObject {
    int id;
    KalmanFilter kf;
    geometry_msgs::PointStamped position;
    double time;
    Eigen::VectorXd velocity;
    Eigen::VectorXd abs_velocity;
    Eigen::VectorXd last_velocity;
    bool is_dynamic;
    int age = 0;
    int dynamic_cooldown = 0;
};

ros::Publisher marker_pub;
ros::Publisher cluster_pub;
std::map<int, TrackedObject> tracked_objects;

double robot_vx = 0.0, robot_vy = 0.0, robot_vz = 0.0;
double imu_angular_x = 0.0, imu_angular_y = 0.0, imu_angular_z = 0.0;

double velocity_thres;
double robot_length, robot_width;
int age_thres;

void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    robot_vx = msg->twist.twist.linear.x;
    robot_vy = msg->twist.twist.linear.y;
    robot_vz = msg->twist.twist.linear.z;
    imu_angular_x = msg->twist.twist.angular.x;
    imu_angular_y = msg->twist.twist.angular.y;
    imu_angular_z = msg->twist.twist.angular.z;
}

void associateCentroids(const std::vector<geometry_msgs::PointStamped>& centroids, double dt) {
    std::vector<bool> matched(centroids.size(), false);
    std::map<int, TrackedObject> new_tracked_objects;

    std::vector<std::vector<double>> cost_matrix(centroids.size(), std::vector<double>(tracked_objects.size(), 0.0));
    std::vector<int> tracked_ids;
    int col = 0;

    for (const auto& [id, obj] : tracked_objects) {
        tracked_ids.push_back(id);
        for (size_t row = 0; row < centroids.size(); ++row) {
            double dx = centroids[row].point.x - obj.position.point.x;
            double dy = centroids[row].point.y - obj.position.point.y;
            double dz = centroids[row].point.z - obj.position.point.z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            double vel_dx = centroids[row].point.x - (obj.position.point.x + obj.velocity[0] * dt);
            double vel_dy = centroids[row].point.y - (obj.position.point.y + obj.velocity[1] * dt);
            double vel_dz = centroids[row].point.z - (obj.position.point.y + obj.velocity[2] * dt);
            double vel_dist = std::sqrt(vel_dx * vel_dx + vel_dy * vel_dy + vel_dz * vel_dz);

            // cost_matrix[row][col] = 0.9 * dist + 0.1 * vel_dist;
            cost_matrix[row][col] = dist;
        }
        col++;
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    double cost = hungarian.Solve(cost_matrix, assignment);

    for (size_t row = 0; row < assignment.size(); ++row) {
        int col = assignment[row];
        if (col >= 0 && col < static_cast<int>(tracked_objects.size()) && cost_matrix[row][col] < 2.0) {
            int best_id = tracked_ids[col];
            TrackedObject& obj = tracked_objects[best_id];

            Eigen::VectorXd measurement(3);
            measurement << centroids[row].point.x, centroids[row].point.y, centroids[row].point.z;

            Eigen::MatrixXd A(6, 6);
            A << 1, 0, 0, dt, 0, 0,
                 0, 1, 0, 0, dt, 0,
                 0, 0, 1, 0, 0, dt,
                 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 1;

            obj.kf.predict(dt, A);
            obj.kf.update(measurement);

            Eigen::VectorXd state = obj.kf.state();
            // obj.velocity = 0.3 * obj.velocity + 0.7 * state.tail(3);
            obj.velocity = state.tail(3);

            obj.abs_velocity(0) = obj.velocity(0) + robot_vx + robot_width*imu_angular_z;
            obj.abs_velocity(1) = obj.velocity(1) + robot_vy + robot_length*imu_angular_z;
            obj.abs_velocity(2) = obj.velocity(2) + robot_vz;

            obj.last_velocity = obj.velocity;
            obj.position = centroids[row];
            obj.time = centroids[row].header.stamp.toSec();
            obj.age++;

            if( (std::hypot(obj.abs_velocity(0), obj.abs_velocity(1)) > velocity_thres) && (obj.age > age_thres) )
            {
                obj.is_dynamic = true;
                obj.dynamic_cooldown = COOLDOWN_FRAMES;
            }else if(obj.dynamic_cooldown > 0)
            {
                obj.dynamic_cooldown--;
            }else
            {
                obj.is_dynamic = false;
            }

            new_tracked_objects[best_id] = obj;
            matched[row] = true;
        }
    }

    int next_id = tracked_objects.empty() ? 0 : tracked_objects.rbegin()->first + 1;
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (!matched[i]) {
            TrackedObject obj;
            obj.id = next_id++;
            obj.age = 1;
            obj.is_dynamic = false;

            Eigen::MatrixXd A(6, 6);
            A << 1, 0, 0, dt, 0, 0,
                 0, 1, 0, 0, dt, 0,
                 0, 0, 1, 0, 0, dt,
                 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 1;
            Eigen::MatrixXd C(3, 6);
            C << 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0;
            Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6) * 1.0;
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * 0.5;
            Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6, 6) * 0.01;

            obj.kf = KalmanFilter(dt, A, C, Q, R, P);
            Eigen::VectorXd x0(6);
            x0 << centroids[i].point.x, centroids[i].point.y, centroids[i].point.z, 0, 0, 0;
            obj.kf.init(centroids[i].header.stamp.toSec(), x0);

            obj.position = centroids[i];
            obj.time = centroids[i].header.stamp.toSec();
            obj.velocity = Eigen::VectorXd::Zero(3);
            obj.abs_velocity = obj.velocity;
            obj.last_velocity = obj.velocity;

            new_tracked_objects[obj.id] = obj;
        }
    }

    tracked_objects = new_tracked_objects;
}

void publishMarkers() {
    visualization_msgs::MarkerArray marker_array;
    double vx, vy, vz;

    for (const auto& [id, obj] : tracked_objects) {

        vx = obj.abs_velocity(0);
        vy = obj.abs_velocity(1);
        vz = obj.abs_velocity(2);

        // Log the position, velocity, and ID of each dynamic obstacle
        ROS_INFO("Dynamic Obstacle ID: %d, Position: [%.2f, %.2f, %.2f], Velocity: [%.2f, %.2f, %.2f], AbsVelocity: [%.2f, %.2f, %.2f]",
            id,
            obj.position.point.x, obj.position.point.y, obj.position.point.z,
            obj.velocity[0], obj.velocity[1], obj.velocity[2],
            vx, vy, vz);

        visualization_msgs::Marker pos_marker;
        pos_marker.header = obj.position.header;
        pos_marker.ns = "tracked_dynamic_objects";
        pos_marker.id = obj.id;
        pos_marker.type = visualization_msgs::Marker::SPHERE;
        pos_marker.action = visualization_msgs::Marker::ADD;
        pos_marker.pose.position = obj.position.point;
        pos_marker.pose.orientation.w = 1.0;
        pos_marker.scale.x = 0.4;
        pos_marker.scale.y = 0.4;
        pos_marker.scale.z = 0.4;
        pos_marker.color.a = 1.0;
        pos_marker.color.r = (obj.is_dynamic) ? 0.0 : 1.0;
        pos_marker.color.g = (obj.is_dynamic) ? 1.0 : 0.0;
        pos_marker.color.b = 0.0;
        pos_marker.lifetime = ros::Duration(0.2);
        marker_array.markers.push_back(pos_marker);

        visualization_msgs::Marker text_marker;
        text_marker.header = obj.position.header;
        text_marker.ns = "tracked_dynamic_ids";
        text_marker.id = obj.id;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position = obj.position.point;
        text_marker.pose.position.z += 0.5;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.x = 0.4;
        text_marker.scale.y = 0.4;
        text_marker.scale.z = 0.4;
        text_marker.color.a = 1.0;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.text = std::to_string(obj.id);
        text_marker.lifetime = ros::Duration(0.2);
        marker_array.markers.push_back(text_marker);

    }

    marker_pub.publish(marker_array);
}

void clusterCallback(const treescope::ClusterArray::ConstPtr& cluster_msg) {
    static ros::Time last_time = ros::Time::now();
    double dt = (cluster_msg->header.stamp - last_time).toSec();
    if (dt <= 0) dt = 0.1;
    last_time = cluster_msg->header.stamp;

    if(cluster_msg->clusters.size() < 1)
        return;

    std::vector<geometry_msgs::PointStamped> centroids;
    centroids.resize(cluster_msg->clusters.size());

    geometry_msgs::PointStamped centroid;
    centroid.header = cluster_msg->header;
    for(int i=0; i < cluster_msg->clusters.size(); i++){
        centroid.point.x = cluster_msg->clusters[i].centroid.x;
        centroid.point.y = cluster_msg->clusters[i].centroid.y;
        centroid.point.z = cluster_msg->clusters[i].centroid.z;
        centroids[i] = centroid;
    }

    associateCentroids(centroids, dt);
    publishMarkers();
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "tracking");
    ros::NodeHandle nh;

    marker_pub = nh.advertise<visualization_msgs::MarkerArray>(nh.getNamespace() + "/tracking/markers", 1);
    cluster_pub = nh.advertise<treescope::ClusterArray>(nh.getNamespace() + "/tracking/clusters", 1);

    ros::NodeHandle nh_priv("~");
    nh_priv.param<double>("Thresholds/Velocity", velocity_thres, 0.5);
    nh_priv.param<int>("Thresholds/Age", age_thres, 5);
    nh_priv.param<double>("RobotDimensions/Lenght", robot_length, 1.0);
    nh_priv.param<double>("RobotDimensions/Width", robot_width, 0.5);

    ros::Subscriber cloud_sub = nh.subscribe(nh.getNamespace() + "/clusters", 1, clusterCallback);
    ros::Subscriber odom_sub = nh.subscribe(nh.getNamespace() + "/fast_limo/state", 1, odomCallback);

    ros::spin();
    return 0;
}