#!/usr/bin/env python

import rospy
import roslib

import cv2 as cv
import numpy as np
import caffe
import os
import sys
import random
import math

import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped as Rect

class HandHheldObjectTracking():
    def __init__(self):
        self.__net = None
        self.__transformer = None
        self.__im_width = None
        self.__im_height = None
        self.__channels = None
        self.__bridge = CvBridge()
        
        self.__weights = rospy.get_param('~pretrained_weights', None)
        self.__model_proto = rospy.get_param('~deployment_prototxt', None)
        self.__device_id = rospy.get_param('device_id', 0)

        self.__rect = None
        self.__batch_size = 1
        self.__templ_rgb = None
        self.__templ_dep = None
        self.__templ_datum = None
        self.__scale = 1.250
        
        ###! temp
        self.__weights = '/home/krishneel/Documents/caffe-tutorials/detection/handheld/snapshot_iter_2966.caffemodel'
        self.__model_proto = '/home/krishneel/Documents/caffe-tutorials/detection/handheld/deploy.prototxt'

        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('TRACKER SETUP SUCCESSFUL')

        self.subscribe()


    def process_rgbd(self, im_rgb, im_dep, rect, scale = 1.5):
        
        ##!crop (build multiple scale)
        rect = self.get_region_bbox(im_rgb, rect, scale)
        x,y,w,h = rect
        im_rgb = im_rgb[y:y+h, x:x+w].copy()
        im_dep = im_dep[y:y+h, x:x+w].copy()
        
        image = im_rgb.copy()

        ##! resize to network input
        im_rgb = cv.resize(im_rgb, (int(self.__im_width), int(self.__im_height)))
        im_dep = cv.resize(im_dep, (int(self.__im_width), int(self.__im_height)))
        
        ##! normalize and encode
        #im_rgb = self.demean_rgb_image(im_rgb)
        im_rgb = im_rgb.astype(np.float32)
        im_rgb /= im_rgb.max()
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        
        im_dep = im_dep.astype(np.float32) \
                 if not im_dep.dtype is str('float32') else  im_dep
        im_dep /= im_dep.max()
        im_dep *= 255.0
        im_dep = im_dep.astype(np.uint8)
        im_dep = cv.applyColorMap(im_dep, cv.COLORMAP_JET)

        im_dep = im_dep.astype(np.float32)
        im_dep /= im_dep.max()
        im_dep = (im_dep - im_dep.min())/(im_dep.max() - im_dep.min())
        
        #! transpose to c, h, w
        im_rgb = im_rgb.transpose((2, 0, 1))
        im_dep = im_dep.transpose((2, 0, 1))

        im_datum = np.zeros((self.__batch_size, self.__channels, \
                             self.__im_height, self.__im_width), np.float32)        
        im_datum[0][0:3, :, :] = im_rgb.copy()
        im_datum[0][3:6, :, :] = im_dep.copy()

        return image, im_datum, rect
        

    def track(self, im_rgb, im_dep):
        caffe.set_device(self.__device_id)
        caffe.set_mode_gpu()

        ##im_dep[im_dep > 1.50] = 100.0
        image, im_datum, rect = self.process_rgbd(im_rgb, im_dep, self.__rect, self.__scale)
        
        if self.__templ_datum is None:
            self.__templ_datum = im_datum.copy()
            self.__scale = 1.250
            
        # cv.namedWindow("depth", cv.WINDOW_NORMAL)
        # cv.imshow("depth", im_dep)
        # cv.waitKey(20)
        # return
        

        self.__net.blobs['target_data'].data[...] = im_datum.copy()
        self.__net.blobs['template_data'].data[...] = self.__templ_datum.copy()

        output = self.__net.forward()
        
        #! self.__templ_datum = im_datum.copy()


        feat = self.__net.blobs['score'].data[0]
        prob = feat[1].copy()
        #prob[prob < 0.5] = 0.0
        #prob[prob >= 0.5] = 1.0
        prob *= 255
        prob = prob.astype(np.uint8)
        
        prob = cv.resize(prob, (rect[2], rect[3]))

        #! get rect
        bbox = self.bounding_rect(prob)
        bbox[0] = rect[0]
        bbox[1] = rect[1]
        
        #self.__rect = bbox

        x, y, w, h = bbox
        cv.rectangle(im_rgb, (x, y), (x+w, h+y), (0, 255, 0), 4)

        res = cv.bitwise_and(image, image,mask = prob)
        
        prob = cv.applyColorMap(prob, cv.COLORMAP_JET)
        res = np.hstack((res, prob))

        cv.namedWindow("region", cv.WINDOW_NORMAL)
        cv.imshow("region", res)

        cv.namedWindow("rgb", cv.WINDOW_NORMAL)
        cv.imshow("rgb", im_rgb)
        if cv.waitKey(1) & 0xFF == ord("q"):
            return

    """
    image callback function
    """
    def callback(self, image_msg, depth_msg):
        im_rgb= self.convert_image(image_msg)
        im_dep = self.convert_image(depth_msg, '32FC1')

        if im_rgb is None or im_dep is None:
            rospy.logwarn('input msg is empty')
            return
            
        im_dep[np.isnan(im_dep)] = 0.0


        if not self.__rect is None:
            self.track(im_rgb, im_dep)
        else:
            rospy.loginfo_throttle(60, 'Object not initialized ...')

    def bounding_rect(self, im_mask):
        x1 = im_mask.shape[1] + 1
        y1 = im_mask.shape[0] + 1
        x2 = 0
        y2 = 0
        for j in xrange(0, im_mask.shape[0], 1):
            for i in xrange(0, im_mask.shape[1], 1):
                if im_mask[j, i] > 0:
                    x1 = i if i < x1 else x1
                    y1 = j if j < y1 else y1
                    x2 = i if i > x2 else x2
                    y2 = j if j > y2 else y2

        rect = np.array([x1, y1, x2 - x1, y2 - y1])
        x, y, w, h = rect
        # bbox = get_region_bbox(im_mask, rect)
        return rect
            

    def get_region_bbox(self, im_rgb, rect, scale = 1.5):
        x, y, w, h = rect
        cx, cy = (x + w/2.0, y + h/2.0)
        s = scale

        nw = int(s * w)
        nh = int(s * h)
        nx = int(cx - nw/2.0)
        ny = int(cy - nh/2.0)

        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-im_rgb.shape[1]) if (nx+nw) > im_rgb.shape[1] else nw
        nh = nh-((ny+nh)-im_rgb.shape[0]) if (ny+nh) > im_rgb.shape[0] else nh

        return np.array([nx, ny, nw, nh])


    def screen_point_callback(self, rect_msg):
        x = rect_msg.polygon.points[0].x
        y = rect_msg.polygon.points[0].y
        w = rect_msg.polygon.points[1].x  - x
        h = rect_msg.polygon.points[1].y - y
        self.__rect = np.array([x, y, w, h])

        rospy.loginfo('Object Rect Received')
        print self.__rect
        

    """
    imagenet mean for demeaning rgb
    """
    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(np.float32)
        im_rgb[:, :, 0] -= np.float32(104.0069879317889)
        im_rgb[:, :, 1] -= np.float32(116.66876761696767)
        im_rgb[:, :, 2] -= np.float32(122.6789143406786)
        #im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb

  
    """
    function to convert ros message to cv image
    """
    def convert_image(self, image_msg, encoding = 'bgr8'):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, encoding)
        except Exception as e:
            print (e)
            return
        
        return cv_img


    def subscribe(self):
        
        rospy.Subscriber('rect', Rect, self.screen_point_callback)


        image_sub = message_filters.Subscriber('image', Image)
        depth_sub = message_filters.Subscriber('depth', Image)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 1)
        ts.registerCallback(self.callback)

    """
    function to load caffe models and pretrained weights
    """
    def load_caffe_model(self):
        rospy.loginfo('LOADING CAFFE MODEL..')
        self.__net = caffe.Net(self.__model_proto, self.__weights, caffe.TEST)
        
        # self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})
        # self.__transformer.set_transpose('data', (2,0,1))
        # self.__transformer.set_raw_scale('data', 1)
        # self.__transformer.set_channel_swap('data', (2,1,0))

        shape = self.__net.blobs['target_data'].data.shape
        self.__channels = shape[1]
        self.__im_height = shape[2]
        self.__im_width = shape[3]

        self.__net.blobs['template_data'].reshape(self.__batch_size, self.__channels, \
                                                  self.__im_height, self.__im_width)
        self.__net.blobs['target_data'].reshape(self.__batch_size, self.__channels, \
                                                self.__im_height, self.__im_width)


    def is_file_valid(self):
        if self.__model_proto is None or \
           self.__weights is None:
            rospy.logfatal('PROVIDE PRETRAINED MODEL! KILLING NODE...')
            return False
        
        is_file = lambda path : os.path.isfile(str(path))
        if  (not is_file(self.__model_proto)) or (not is_file(self.__weights)):
            rospy.logfatal('NOT SUCH FILES')
            return False

        return True


def main(argv):
    try:
        rospy.init_node('handheld_object_tracking', anonymous = True)
        hhot = HandHheldObjectTracking()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass


if __name__ == '__main__':
    main(sys.argv)
