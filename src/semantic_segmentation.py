#!/usr/bin/env python3

import os
import sys

import csv
import torch
import numpy 
import scipy.io
import PIL.Image
import torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, Imu
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import open3d as o3d
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import rospy
import matplotlib as mpl
import matplotlib.cm as cm
import open3d as o3d
import PIL.Image as pilImage
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage

from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth
import networkslite
import PIL.Image as pilImage
from torchvision import transforms, datasets

# from realsense_camera.msg import BowQueries
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros

from tf2_msgs.msg import TFMessage

from semantic_segmentation_publisher.msg import BowQueries  # Replace 'your_package_name' with the actual name of your package
from semantic_segmentation_publisher.msg import BowQuery  # Ensure you import all the necessary custom messages
from semantic_segmentation_publisher.msg import BowVector



from layers import disp_to_depth
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class SemanticSegmentationPublisher:

    def __init__(self):
        self.bridge = CvBridge()
        self.seq = 0
        self.imu_seq = 0
        self.device = "cuda"
        
        self.script_dir = os.path.dirname(os.path.realpath(__file__))


        rospy.init_node('semantic_segmentation_publisher_node', anonymous=True)

        # Get the parameters from the launch file
        # self.model_type = rospy.get_param('~model_type', "hrnetv2+c1")
        # self.depth_model_type = rospy.get_param('~depth_model_type', 'lite_mono')

        self.model_type = rospy.get_param('~model_type')
        self.depth_model_type = rospy.get_param('~depth_model_type')
        rospy.loginfo(self.model_type)
        rospy.loginfo(self.depth_model_type)

        # Topics
        # self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        # self.depth_info_topic = rospy.get_param('~depth_info_topic', '/tesse/depth_cam/mono/camera_info')
        # self.depth_image_topic = rospy.get_param('~depth_image_topic', 'camera/aligned_depth_to_color/image_raw')
        # self.rgb_topic = rospy.get_param('~rgb_topic', '/tesse/left_cam/rgb/image_raw')
        # self.rgb_info_topic = rospy.get_param('~rgb_info_topic', '/tesse/left_cam/camera_info')
        # self.depth_topic = rospy.get_param('~depth_topic', '/tesse/depth_cam/mono/image_raw')
        # self.seg_rgb_topic = rospy.get_param('~seg_rgb_topic', '/tesse/seg_cam/rgb/image_raw')

        self.image_topic = rospy.get_param('~image_topic')
        self.depth_info_topic = rospy.get_param('~depth_info_topic')
        self.depth_image_topic = rospy.get_param('~depth_image_topic')
        self.rgb_topic = rospy.get_param('~rgb_topic')
        self.rgb_info_topic = rospy.get_param('~rgb_info_topic')
        self.depth_topic = rospy.get_param('~depth_topic')
        self.seg_rgb_topic = rospy.get_param('~seg_rgb_topic')
        self.odometry_topic = rospy.get_param('~odometry_topic')
        self.odom_publish_topic = rospy.get_param('~odom_publish_topic')
        self.bow_vector = rospy.get_param('~bow_vector')
        self.bow_vector_pub_topic = rospy.get_param('~bow_vector_pub_topic')


        # Publishers
        self.pub_rgb_image = rospy.Publisher(self.rgb_topic, Image, queue_size=100)
        self.pub_rgb_camera_info = rospy.Publisher(self.rgb_info_topic, CameraInfo, queue_size=100)
        self.pub_depth_camera_info = rospy.Publisher(self.depth_info_topic, CameraInfo, queue_size=100)
        self.pub_depth_image = rospy.Publisher(self.depth_topic, Image, queue_size=10)
        self.pub_semantic_image = rospy.Publisher(self.seg_rgb_topic, Image, queue_size=100)
        self.odometry_publish = rospy.Publisher(self.odom_publish_topic, Odometry, queue_size=100)
        # self.imu_publish = rospy.Publisher('/tesse/imu', Imu, queue_size=100)
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=10)
        self.static_tf_pub = rospy.Publisher('/tf_static', TFMessage, queue_size=10, latch=True)
        self.bow_queries_pub = rospy.Publisher(self.bow_vector_pub_topic, BowQueries, queue_size=10)


        # Subscribers
        # self.depth_image_sub = Subscriber(self.depth_image_topic, Image)
        # self.color_image_sub = Subscriber(self.image_topic, Image)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.odometry_topic, Odometry, self.odometry_callback)
        rospy.Subscriber(self.bow_vector, BowQueries, self.bow_queries_callback)
        # rospy.Subscriber('/camera/imu', Imu, self.imu_callback)

        # # Time Synchronizer
        # self.sync = ApproximateTimeSynchronizer([self.depth_image_sub, self.color_image_sub], queue_size=10, slop=0.1)
        # self.sync.registerCallback(self.image_depth_callback)

        rospy.loginfo("Waiting for network initialize ...!")

        # Network Initialization
        self.segmentation_module, self.image_transform, self.colors, self.names = self.semantic_segmentation_init(self.model_type)
        self.depth_encoder, self.depth_decoder = self.depth_network_init(self.depth_model_type)

        rospy.loginfo("Network Initialized and start the rosbag ...!")

        # self.publish_static_transforms()

    def bow_queries_callback(self, msg):
        # This is where you can process the message if needed
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link_gt"  

        rospy.loginfo("Changed frame_id to: %s", msg.header.frame_id)


        # For demonstration, we're just republishing the same message
        self.bow_queries_pub.publish(msg)

    def imu_callback(self, msg):

        msg.header.seq = self.imu_seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link_gt"

        self.imu_publish.publish(msg)

        self.imu_seq += 1

    def publish_static_transforms(self):

        # Create a publisher for the /tf_static topic

        # 
        transforms = []
        
        # transform frame needed for RGB image, Depth image, semantic image

        transform_0 = TransformStamped()
        transform_0.header.seq = 0
        transform_0.header.stamp = rospy.Time.now()
        transform_0.header.frame_id = 'base_link_gt'
        transform_0.child_frame_id = 'left_cam'
        transform_0.transform.translation.x = 0.0
        transform_0.transform.translation.y = 0.0
        transform_0.transform.translation.z = 0.0
        transform_0.transform.rotation.x = 0.0
        transform_0.transform.rotation.y = 0.0
        transform_0.transform.rotation.z = 0.0
        transform_0.transform.rotation.w = 1.0
        transforms.append(transform_0)

        transform_2 = TransformStamped()
        transform_2.header.seq = 0
        transform_2.header.stamp = rospy.Time.now()
        transform_2.header.frame_id = 'base_link_gt'
        transform_2.child_frame_id = 'left_cam'
        transform_2.transform.translation.x = 0.0
        transform_2.transform.translation.y = 0.0
        transform_2.transform.translation.z = 0.0
        transform_2.transform.rotation.x = 0.0
        transform_2.transform.rotation.y = 0.0
        transform_2.transform.rotation.z = 0.0
        transform_2.transform.rotation.w = 1.0
        transforms.append(transform_2)

        transform_3 = TransformStamped()
        transform_3.header.seq = 0
        transform_3.header.stamp = rospy.Time.now()
        transform_3.header.frame_id = 'base_link_gt'
        transform_3.child_frame_id = 'left_cam'
        transform_3.transform.translation.x = 0.0
        transform_3.transform.translation.y = 0.0
        transform_3.transform.translation.z = 0.0
        transform_3.transform.rotation.x = 0.0
        transform_3.transform.rotation.y = 0.0
        transform_3.transform.rotation.z = 0.0
        transform_3.transform.rotation.w = 1.0
        transforms.append(transform_3)

        # Create a TFMessage and publish it
        tf_message = TFMessage(transforms=transforms)
        self.static_tf_pub.publish(tf_message)

    def odometry_callback(self, msg):
        
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.child_frame_id = "base_link_gt"

        pose = msg.pose.pose

        # TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = "world"  # Parent frame
        transform.child_frame_id = "base_link_gt"  # Child frame

        #  translation
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z

        #  rotation
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w

        #  transform
        tf_message = TFMessage(transforms=[transform])
        self.tf_pub.publish(tf_message)

        self.odometry_publish.publish(msg)
        
    def colorEncode(self, labelmap, colors, mode='RGB'):
        labelmap = labelmap.astype('int')
        labelmap_rgb = numpy.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=numpy.uint8)
        for label in numpy.unique(labelmap):
            if label < 0 or label >= len(colors):  # Skip invalid or out-of-range labels
                continue
            labelmap_rgb += (labelmap == label)[:, :, numpy.newaxis] * colors[label]

        if mode == 'BGR':
            return labelmap_rgb[:, :, ::-1]
        else:
            return labelmap_rgb

    def visualize_result(self, img, pred, names, colors, index=None):
        predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
 
        if index is not None:
            pred = pred.copy()
            pred[pred != predicted_classes] = -1

        pred_color = colorEncode(pred, colors).astype(numpy.uint8)
        im_vis = numpy.concatenate((img, pred_color), axis=1)

        plt.ion()  # Turn on interactive mode
        plt.imshow(im_vis)
        plt.draw()
        plt.pause(0.01)  # Add a short pause to allow time for the window to update
        plt.ioff() 

        # PIL.Image.fromarray(im_vis).show()
        print('image_generated')

    def distdepth_callback(self, cv_image, common_header):


        with torch.no_grad():

            original_size = (cv_image.shape[0], cv_image.shape[1])

            raw_img = numpy.transpose(cv_image, (2, 0, 1))
            input_image = torch.from_numpy(raw_img).float().to(self.device)
            input_image = (input_image / 255.0).unsqueeze(0)

            # Resize to input size
            input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )
            features = self.depth_encoder(input_image)
            outputs = self.depth_decoder(features)

            out = outputs[("out", 0)]

            # Resize to original size
            out_resized = torch.nn.functional.interpolate(
                out, original_size, mode="bilinear", align_corners=False
            )

            # Convert disparity to depth
            depth = output_to_depth(out_resized, 0.1, 10)
            # metric_depth = depth.detach().cpu().numpy().squeeze()
            metric_depth = depth.detach().cpu().numpy().squeeze()

            depth_image = metric_depth.astype(numpy.float32)  # Ensure the array is in float32 format
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
            ros_depth_image_msg.header = common_header

            
            self.pub_depth_image.publish(ros_depth_image_msg)

    def semantic_segmentation_init(self,model_type):

        # Construct file paths dynamically
        data_dir = os.path.join(self.script_dir, "data")
        ckpt_dir = os.path.join(self.script_dir, "ckpt")

        colors_path = os.path.join(data_dir, "ade150_config.mat")
        csv_path = os.path.join(data_dir, "object150_info.csv")

        # Load colors
        colors = scipy.io.loadmat(colors_path)['colors']

        # Load object names
        names = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]

        # Model selection
        if model_type == 'resnet50dilated+ppm_deepsup':
            encoder_path = os.path.join(ckpt_dir, "ade20k-resnet50dilated-ppm_deepsup", "encoder_epoch_20.pth")
            decoder_path = os.path.join(ckpt_dir, "ade20k-resnet50dilated-ppm_deepsup", "decoder_epoch_20.pth")

            net_encoder = ModelBuilder.build_encoder(arch='resnet50dilated', fc_dim=2048, weights=encoder_path)
            net_decoder = ModelBuilder.build_decoder(arch='ppm_deepsup', fc_dim=2048, num_class=150, weights=decoder_path, use_softmax=True)

            rospy.loginfo("resnet50dilated+ppm_deepsup Initialized")

        elif model_type == 'hrnetv2+c1':
            encoder_path = os.path.join(ckpt_dir, "ade20k-hrnetv2-c1", "encoder_epoch_30.pth")
            decoder_path = os.path.join(ckpt_dir, "ade20k-hrnetv2-c1", "decoder_epoch_30.pth")

            net_encoder = ModelBuilder.build_encoder(arch='hrnetv2', fc_dim=720, weights=encoder_path)
            net_decoder = ModelBuilder.build_decoder(arch='c1', fc_dim=720, num_class=150, weights=decoder_path, use_softmax=True)

            rospy.loginfo("hrnetv2+c1 Initialized")

        # Set up segmentation module
        crit = torch.nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        segmentation_module.eval()
        segmentation_module.cuda()

        # Image transform pipeline
        image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((412, 600)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return segmentation_module, image_transform, colors, names
        
    def depth_network_init(self,model_type):
    
    		
        ckpt_dir = os.path.join(self.script_dir, "ckpt")


        distdepth_dir = os.path.join(ckpt_dir, "distdepth_pretrainedmodel")
        lite_mono_dir = os.path.join(ckpt_dir, "litepretrainedmodel", "pretrained_models")


        if model_type == 'distdepth':

            encoder_path = os.path.join(distdepth_dir, "encoder.pth")
            decoder_path = os.path.join(distdepth_dir, "depth.pth")

            depth_encoder = ResnetEncoder(152, False)
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
            depth_encoder.load_state_dict(filtered_dict_enc)
            depth_encoder.to(self.device)
            depth_encoder.eval()

            depth_decoder = DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
            loaded_dict = torch.load(decoder_path, map_location=self.device)
            depth_decoder.load_state_dict(loaded_dict)

            depth_decoder.to(self.device)
            depth_decoder.eval()

            rospy.loginfo("DistDepth network initialized ...!")

        elif model_type == 'lite_mono':

            encoder_path = os.path.join(lite_mono_dir, "encoder.pth")
            decoder_path = os.path.join(lite_mono_dir, "depth.pth")

            depth_encoder = torch.load(encoder_path, map_location=torch.device('cuda'))
            depth_decoder = torch.load(decoder_path, map_location=torch.device('cuda'))

            feed_height = depth_encoder['height']
            feed_width = depth_encoder['width']

            depth_encoder = networkslite.LiteMono(model='lite-mono', height=feed_height, width=feed_width)
            model_dict = depth_encoder.state_dict()
            depth_encoder.load_state_dict({k: v for k, v in depth_encoder.state_dict().items() if k in model_dict})
            depth_encoder.to('cuda')
            depth_encoder.eval()

            depth_decoder = networkslite.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(3))
            depth_model_dict = depth_decoder.state_dict()
            depth_decoder.load_state_dict({k: v for k, v in depth_decoder.state_dict().items() if k in depth_model_dict})
            depth_decoder.to('cuda')
            depth_decoder.eval()

            rospy.loginfo("Lite-Mono network initialized ...!")

        return depth_encoder, depth_decoder

    def lite_mono_callback(self,cv_image,common_header):
        with torch.no_grad():

            original_height, original_width  = cv_image.shape[:2]
            # input_image = input_image.resize((feed_width, feed_height), pilImage.LANCZOS)
            input_image = transforms.ToTensor()(cv_image).unsqueeze(0).to(self.device)


            # Get depth features and outputs

            # input_image = input_image.to(self.device)
            features = self.depth_encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]

            # Resize the disparity map to the original image size

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # output_name = os.path.splitext(os.path.basename(image_path))[0]


            # Convert disparity to depth
            scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)

            scaled_disp_np = scaled_disp.squeeze().cpu().numpy()
            depth_np = depth.squeeze().cpu().numpy()


            disp_resized_np = disp_resized.squeeze().cpu().numpy()


            depth_image = scaled_disp_np.astype(numpy.float32)  # Ensure the array is in float32 format
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
            ros_depth_image_msg.header = common_header

            
            self.pub_depth_image.publish(ros_depth_image_msg)
 
    def semantic_segmentation_function(self, common_header, cv_color_image):
            
            pil_image = PIL.Image.fromarray(cv_color_image)
            original_size = pil_image.size


            img_original = numpy.array(pil_image)
            img_data = self.image_transform(pil_image)
            singleton_batch = {'img_data': img_data[None].cuda()}
            output_size = img_data.shape[1:]

            with torch.no_grad():
                scores = self.segmentation_module(singleton_batch, segSize=output_size)
            
            _, pred = torch.max(scores, dim=1)
            pred = pred.cpu()[0].numpy()
            pred = cv2.resize(pred, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
            pred_color = colorEncode(pred, self.colors).astype(numpy.uint8)
            

            ros_seg_image_msg = Image()
            ros_seg_image_msg.header, ros_seg_image_msg.encoding  = common_header, "rgb8"
            ros_seg_image_msg.height, ros_seg_image_msg.width = pred_color.shape[0], pred_color.shape[1]
            ros_seg_image_msg.step, ros_seg_image_msg.data = pred_color.shape[1]*3,  pred_color.tobytes()

            self.pub_semantic_image.publish(ros_seg_image_msg)

    def image_callback(self, color_msg):
        try:
            # Convert ROS Image messages to OpenCV images

            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

            common_header = color_msg.header
            common_header.seq = self.seq
            common_header.stamp = rospy.Time.now()
            common_header.frame_id = "left_cam"

            color_msg.header.frame_id = "left_cam"

            republish_camera_raw_image = color_msg

            self.semantic_segmentation_function(common_header,cv_color_image)

            if self.depth_model_type == 'distdepth':
                self.distdepth_callback(cv_color_image, common_header)
            elif self.depth_model_type == 'lite_mono':
                self.lite_mono_callback(cv_color_image, common_header)

            self.camera_info_callback(common_header)

            self.pub_rgb_image.publish(republish_camera_raw_image)

            # self.pub_depth_image.publish(ros_depth_image_msg)

            self.seq += 1

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def image_depth_callback(self, depth_msg, color_msg):
        try:
            # Convert ROS Image messages to OpenCV images

            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

            common_header = color_msg.header
            common_header.seq = self.seq
            common_header.stamp = rospy.Time.now()
            common_header.frame_id = "left_cam"

            color_msg.header.frame_id = "left_cam"

            republish_camera_raw_image = color_msg

            self.semantic_segmentation_function(common_header,cv_color_image)

            # realsense depth image ---------
            # depth_msg.header = common_header

            # ros_depth_image_msg = depth_msg
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(cv_depth_image, "16UC1")
            ros_depth_image_msg.header = common_header
            #-------------------------------

            self.camera_info_callback(common_header)

            self.pub_rgb_image.publish(republish_camera_raw_image)

            self.pub_depth_image.publish(ros_depth_image_msg)

            self.seq += 1

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def camera_info_callback(self, header): 


        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.height = 1100
        camera_info.width = 1600
        camera_info.distortion_model = "radial-tangential"
        camera_info.D = [-0.14739918980183578, 0.13397075154766996, 0.00022546630381360626, 0.0003564432549001406]
        camera_info.K = [1317.9744707808113, 0, 815.6206980533643, 0, 1317.9744707808113, 557.6887328794995, 0, 0, 1]
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        camera_info.P = [1317.9744707808113, 0, 815.6206980533643, 0, 0, 1317.9744707808113, 557.6887328794995, 0, 0, 0, 1, 0]
        camera_info.binning_x = 0
        camera_info.binning_y = 0
        camera_info.roi.x_offset = 0
        camera_info.roi.y_offset = 0
        camera_info.roi.height = 0
        camera_info.roi.width = 0
        camera_info.roi.do_rectify = False

        self.pub_rgb_camera_info.publish(camera_info)

        self.pub_depth_camera_info.publish(camera_info)

    def main(self):
        rate = rospy.Rate(1) 

        rospy.spin()

if __name__ == "__main__":
    publisher = SemanticSegmentationPublisher()
    publisher.main()
