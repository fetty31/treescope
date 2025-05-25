#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

class ChangeTypeImage {
public:
ChangeTypeImage() : it_(nh_) {

        sub_ = it_.subscribe("/camera/image_raw", 1, &ChangeTypeImage::callback, this);
        pub_ = it_.advertise("/camera/image_dtype_raw", 1);

    }

private:
    void callback(const sensor_msgs::Image::ConstPtr& msg) {
        
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            pub_.publish(cv_ptr->toImageMsg());
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }

    }

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_type_change");
    ChangeTypeImage dtype;
    ros::spin();
    return 0;
}