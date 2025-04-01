#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageResizer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_resizer', anonymous=True)

        # Create a CvBridge object
        self.bridge = CvBridge()

        # Subscribe to the input image topic
        self.image_sub = rospy.Subscriber('/camera_array/cam0/image_raw', Image, self.image_callback)

        # Publisher for the resized image
        self.image_pub = rospy.Publisher('/camera_array/cam0/image_raw_1', Image, queue_size=1000)

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize the image to 800x550
            # resized_image = cv2.resize(cv_image, (800, 550))

            # Convert the OpenCV image back to a ROS Image message
            resized_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            resized_msg.header = msg.header

            # Publish the resized image
            self.image_pub.publish(resized_msg)

            print("Image published ..")

        except Exception as e:
            rospy.logerr("Failed to process image: %s" % str(e))

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        # Create an instance of the ImageResizer class and start it
        image_resizer = ImageResizer()
        image_resizer.start()
    except rospy.ROSInterruptException:
        pass
