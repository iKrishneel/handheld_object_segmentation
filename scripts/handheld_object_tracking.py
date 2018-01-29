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
import time

import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PolygonStamped as Rect

(CV_MAJOR, CV_MINOR, _) = cv.__version__.split(".")

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

        self.__scales = np.array([1.250], dtype = np.float32)

        self.__rect = None
        self.__batch_size = int(self.__scales.shape[0])

        #! save previous data
        self.__prev_roi = None
        self.__prev_rgb = None
        self.__prev_dep = None

        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('TRACKER SETUP SUCCESSFUL')

        self.__im_datum = np.zeros((self.__batch_size, self.__channels, \
                                    self.__im_height, self.__im_width), np.float32)
        self.__templ_datum = np.zeros((self.__batch_size, self.__channels, \
                                       self.__im_height, self.__im_width), np.float32)

        self.__image_pub = rospy.Publisher('/probability_map', Image, queue_size = 1)
        self.__image_pub2 = rospy.Publisher('/region', Image, queue_size = 1)
        self.__rect_pub = rospy.Publisher('/object_rect', Rect, queue_size = 1)

        self.subscribe()

    def normalize_data(self, im_rgb, im_dep):
        ##! normalize and encode
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

        return im_rgb, im_dep

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
        
        #! transpose to c, h, w
        im_rgb = im_rgb.transpose((2, 0, 1))
        im_dep = im_dep.transpose((2, 0, 1))

        return im_rgb, im_dep, image, rect

        
    def track(self, im_rgb, im_dep, header = None):
        caffe.set_device(self.__device_id)
        caffe.set_mode_gpu()

        dist_mask_thresh = 4
        dist_mask_thresh *= 1000.0 if im_dep.max() > 1000.00 else 1.0
        
        im_dep[im_dep > dist_mask_thresh] = 0.0  #! mask depth
        im_nrgb, im_ndep = self.normalize_data(im_rgb, im_dep) #! normalize data

        if self.__prev_dep is None or self.__prev_rgb is None or self.__prev_roi is None:
            self.__prev_rgb = im_nrgb.copy()
            self.__prev_dep = im_ndep.copy()
            self.__prev_roi = self.__rect

        crop_rects = []
        for index, scale in enumerate(self.__scales):
            in_rgb, in_dep, image, rect = self.process_rgbd(im_nrgb, im_ndep, \
                                                            self.__rect.copy(), scale)
            self.__im_datum[index][0:3, :, :] = in_rgb.copy()
            self.__im_datum[index][3:6, :, :] = in_dep.copy()
            crop_rects.append(rect)
            
            ##! template cropping
            in_rgb, in_dep, image, prect = self.process_rgbd(self.__prev_rgb, self.__prev_dep, \
                                                             self.__prev_roi.copy(), scale)
            self.__templ_datum[index][0:3, :, :] = in_rgb.copy()
            self.__templ_datum[index][3:6, :, :] = in_dep.copy()
            
        self.__net.blobs['target_data'].data[...] = self.__im_datum.copy()
        self.__net.blobs['template_data'].data[...] = self.__templ_datum.copy()

        output = self.__net.forward()

        update_model = True
        tmp_rect = self.__rect.copy()

        probability_map = []
        
        for index in xrange(0, self.__batch_size, 1):
            feat = self.__net.blobs['score'].data[index]
            prob = feat[1].copy()
            prob *= 255
            prob = prob.astype(np.uint8)
            
            rect = crop_rects[index]
            prob = cv.resize(prob, (rect[2], rect[3]))
            probability_map.append(prob)

        for index, prob in enumerate(probability_map):
            ##!
            debug = False
            if debug:
                im_prob = cv.applyColorMap(prob, cv.COLORMAP_JET)
                cv.imshow('prob_' + str(index), im_prob)
                cv.waitKey(20)
            ##!
            
            kernel = np.ones((7, 7), np.uint8)
            prob = cv.erode(prob, kernel, iterations = 1)

            prob = cv.GaussianBlur(prob, (5, 5), 0)
            _, prob = cv.threshold(prob, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            rect = crop_rects[index]
            bbox = self.create_mask_rect(prob, rect)
            if bbox is None:
                update_model = False
                return
            
            bbox = np.array(bbox, dtype=np.int)
            bbox[0] += rect[0]
            bbox[1] += rect[1]

            #! enlarge by padding
            bbox = self.get_bbox(im_rgb, bbox, 10)

            im_mask = np.zeros((im_rgb.shape[0], im_rgb.shape[1]), dtype = np.uint8)
            x,y,w,h = crop_rects[index]
            im_mask[y:y+h, x:x+w] = prob

            # area_ratio = float(bbox[2] * bbox[3]) / float(self.__rect[2] * self.__rect[3])
            # if area_ratio > 0.5 and area_ratio < 2 :
            self.__rect = bbox.copy()
            
            x, y, w, h = bbox
            cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(h+y)), (0, 255, 0), 4)

            #! test
            kernel = np.ones((9, 9), np.uint8)
            im_mask = cv.dilate(im_mask, kernel, iterations = 1)

            # im_mask1 = cv.cvtColor(im_mask1, cv.COLOR_GRAY2BGR)
            # cv.addWeighted(im_rgb, 0.5, im_mask1, 0.5, 0, im_rgb)
            # cv.imshow("img", im_rgb)
            # cv.waitKey(3)
            #!
            
            ##! remove incorrect mask by depth masking
            im_mask2 = np.zeros(im_mask.shape, np.uint8)
            im_mask2[y:y+h, x:x+w] = im_mask[y:y+h, x:x+w]
            im_mask = self.depth_mask_filter(im_dep, im_mask2)
            
            ##! reduce mask by scale
            prob_msg = self.__bridge.cv2_to_imgmsg(im_mask, "mono8")
            prob_msg.header = header
            self.__image_pub.publish(prob_msg)
            self.__image_pub2.publish(self.__bridge.cv2_to_imgmsg(im_rgb, "bgr8"))

            rect_msg = Rect()
            rect_msg.header = header
            pt = Point32()
            pt.x = bbox[0]
            pt.y = bbox[1]
            rect_msg.polygon.points.append(pt)
            pt = Point32()
            pt.x = bbox[2] + bbox[0]
            pt.y = bbox[3] + bbox[1]
            rect_msg.polygon.points.append(pt)
            self.__rect_pub.publish(rect_msg)
            
        if update_model:
            self.__prev_rgb = im_nrgb.copy()
            self.__prev_dep = im_ndep.copy()
            self.__prev_roi = tmp_rect


    def depth_mask_filter(self, im_dep, im_mask, max_dist = 1.5):
        im_mask[im_dep > max_dist] = 0
        return im_mask
                
    def create_mask_rect(self, im_gray, rect):  #! rect used for cropping
        if len(im_gray.shape) is None:
            print 'ERROR: Empty input mask'
            return

        if CV_MAJOR < str(3):
            contour, hier = cv.findContours(im_gray.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        else:
            im, contour, hier = cv.findContours(im_gray.copy(), cv.RETR_CCOMP, \
                                                cv.CHAIN_APPROX_SIMPLE)

        prev_center = np.array([self.__rect[0] + self.__rect[2]/2.0, \
                                self.__rect[1] + self.__rect[3]/2.0])

        use_area = True
        max_area = 0
        min_distance = sys.float_info.max
        index = -1
        for i, cnt in enumerate(contour):
            if not use_area:
                box = cv.boundingRect(contour[index])
                center = np.array([box[0] + rect[0] + box[2] / 2.0, \
                                   box[1] + rect[1] + box[3] / 2.0])
                distance = np.linalg.norm(prev_center - center)

                if distance < min_distance:
                    min_distance = distance
                    index = i

            a = cv.contourArea(cnt)
            if max_area < a:
                max_area = a
                index = i
            
        rect = cv.boundingRect(contour[index]) if index > -1 else None
        return rect

    def get_bbox(self, im_rgb, rect, pad = 8):
        x, y, w, h = rect

        nx = int(x - pad)
        ny = int(y - pad)
        nw = int(w  + (2 * pad))
        nh = int(h  + (2 * pad))
        
        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-im_rgb.shape[1]) if (nx+nw) > im_rgb.shape[1] else nw
        nh = nh-((ny+nh)-im_rgb.shape[0]) if (ny+nh) > im_rgb.shape[0] else nh
        
        return np.array([nx, ny, nw, nh])
        
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
            self.track(im_rgb, im_dep, image_msg.header)

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
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 2, 1)
        ts.registerCallback(self.callback)

    """
    function to load caffe models and pretrained weights
    """
    def load_caffe_model(self):
        rospy.loginfo('LOADING CAFFE MODEL..')
        self.__net = caffe.Net(self.__model_proto, self.__weights, caffe.TEST)

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
