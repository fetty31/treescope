#!/usr/bin/env python3
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import torch
from ultralytics import YOLO
import tf
import cv2
import numpy as np
from utils import pointcloud2_to_xyz_array
from tf.transformations import quaternion_matrix

from treescope.msg import YoloDetect2D, YoloDetect2DArray
from treescope.msg import YoloDetect2DMask, YoloDetect2DMaskArray

class YoloDetector:

  def __init__(self):

    self.image_pub = rospy.Publisher("~detections_img", Image, queue_size = 1)
    self.yolo_detection_2d_array_pub = rospy.Publisher("~detections", YoloDetect2DArray, queue_size = 1)
    self.yolo_detection_2d_mask_array_pub = rospy.Publisher("~detections_mask", YoloDetect2DMaskArray, queue_size = 1)

    self.bridge = CvBridge()

    self.landmark_detector = YOLO(rospy.get_param("~ml_models_path")) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.landmark_detector.to(device)
    self.br = tf.TransformBroadcaster()
    self.listener = tf.TransformListener()

    self.input_is_raw = rospy.get_param('~input_is_raw', True)

    if self.input_is_raw:
      self.image_sub = message_filters.Subscriber("~input_image", Image)
    else:
      self.image_sub = message_filters.Subscriber("~input_image", CompressedImage)
    self.camera_info_sub = message_filters.Subscriber("~camera_info", CameraInfo)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.camera_info_sub], 1, 1)
    self.ts.registerCallback(self.callback)

    self.publish_img = rospy.get_param('~publish_img', True) 

  def create_lm(self, class_id, class_type, conf, xyxy):
    yolo_msg = YoloDetect2D()

    yolo_msg.class_id = class_id
    yolo_msg.class_type = class_type
    yolo_msg.conf = conf

    yolo_msg.x_bottom = xyxy[0]
    yolo_msg.y_bottom = xyxy[1]
    yolo_msg.x_top = xyxy[2]
    yolo_msg.y_top = xyxy[3]

    return yolo_msg

  def create_lm_mask(self, class_id,class_type, conf, indexes):
    yolo_msg = YoloDetect2DMask()

    yolo_msg.class_id = class_id
    yolo_msg.class_type = class_type
    yolo_msg.conf = conf

    yolo_msg.indexes_x=indexes[:,1]
    yolo_msg.indexes_y=indexes[:,0]

    return yolo_msg
 
  def callback(self, image_data, camera_info_data):


    # Try reading Img data
    if self.input_is_raw:
      try:
        cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
      except CvBridgeError as e:
        print(e)
        return
    else:
      #### direct conversion to CV2 ####
      np_arr = np.fromstring(image_data.data, np.uint8)
      cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
    
    # Get Detections from image
    results = self.landmark_detector(cv_image,verbose=False)[0]

    # Create list to store detection msgs
    yolo_landmarks_2d_array = []
    yolo_landmarks_2d_mask_array = []

    # # Proceed only if there are semantic objects properly detected
    # if results.masks is not None:

    #   # Evaluate each invidial mask individually
    #   for count, (box, mask) in enumerate(zip(results.boxes, results.masks)):

    #     # Getting the class ID of the given segmentation mask
    #     names = self.landmark_detector.names
    #     class_type = box.cpu().cls.numpy()[0].astype(int)
    #     class_id = names[box.cpu().cls.numpy()[0].astype(int)]

    #     # Getting the confidence of a given ID of a prediction
    #     conf = box.cpu().conf.numpy()[0]

    #     # Getting the Bounding Box xyxy
    #     box_coor = box.cpu().xyxy.numpy()[0].astype(int)

    #     # Create detections msg
    #     yolo_landmarks_2d_array.append(self.create_lm(class_id, class_type, conf, box_coor))

    #     # Resize segmentation mask to input image size
    #     seg_mask = cv2.resize(mask.cpu().data[0].numpy(), (cv_image.shape[1], cv_image.shape[0]))

    #     # Find the x, y coordinates of the pixels different to 0 (segmented objects)
    #     seg_idx = np.argwhere(seg_mask != 0)

    #     # Create mask detections msg
    #     yolo_landmarks_2d_mask_array.append(self.create_lm_mask(class_id, class_type, conf, seg_idx))
    #     if self.publish_img:
    #       # Color the masks for visualization purposes
    #       cv_image[seg_idx[:, 0], seg_idx[:, 1], :] =  np.array([255, 0, 0]) #np.random.randint(0, 255, (1, 3)) 

    for count, box in enumerate(results.boxes):

      # Detection id
      class_id = box.cls.cpu().detach().numpy()[0].astype(int)

      # CODE BLOCK TO DISCARD UNWANTED DETECTIONS SUCH A DYNAMIC OBJECTS
      if class_id == 0: # DISCARD DETECTED HUMANS
        continue
      # else if ... 

      # Detection Bounding Box
      image_points_2D = box.xyxy.cpu().detach().numpy()[0].astype(int)

      print("printing", class_id, image_points_2D)

      if self.publish_img:
        # Visualize Bounding Boxes and Corners
        cv2.rectangle(cv_image, (image_points_2D[0], image_points_2D[1]), (image_points_2D[2], image_points_2D[3]), (0, 255, 0), 2)

    landmark_detection_2d_array_msg = YoloDetect2DArray()
    landmark_detection_2d_array_msg.header = image_data.header
    landmark_detection_2d_array_msg.detections = yolo_landmarks_2d_array
    self.yolo_detection_2d_array_pub.publish(landmark_detection_2d_array_msg)

    landmark_detection_2d_mask_array_msg = YoloDetect2DMaskArray()
    landmark_detection_2d_mask_array_msg.header = image_data.header
    landmark_detection_2d_mask_array_msg.detections = yolo_landmarks_2d_mask_array
    self.yolo_detection_2d_mask_array_pub.publish(landmark_detection_2d_mask_array_msg)

    # Publish Image
    if self.publish_img:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main(args):
  rospy.init_node('yolo_detector_node', anonymous=True)
  detector = YoloDetector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
