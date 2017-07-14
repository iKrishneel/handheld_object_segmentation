#!/usr/bin/env python

import sys
import os
import rospy
import rosbag
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped as Rect
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

class ImageRectWriter:
    def __init__(self):
        self.__write_path = '/home/krishneel/Documents/datasets/handheld_objects2/'
        self.__text_filename = 'train.txt'
        self.__obj_name = 'kleenx/'
        self.__label = 3
        self.__bridge = CvBridge()
        self.__counter = 0

        self.__write_rect = True
        
        if not os.path.exists(str(self.__write_path + self.__obj_name)):
            os.makedirs(str(self.__write_path + self.__obj_name))
            
        self.__img_path = self.__write_path + self.__obj_name + 'image/'
        if not os.path.exists(str(self.__img_path)):
            os.makedirs(str(self.__img_path))
            
        self.__mask_path = self.__write_path + self.__obj_name + 'mask/'
        if not os.path.exists(str(self.__mask_path)):
            os.makedirs(str(self.__mask_path))
            
        self.subscribe()

    def callback(self, image_msg, mask_msg):
        cv_img = self.convert_image(image_msg)
        mask_img = self.convert_image(mask_msg, '8UC1')

        # x = rect_msg.polygon.points[0].x
        # y = rect_msg.polygon.points[0].y
        # w = rect_msg.polygon.points[1].x - x
        # h = rect_msg.polygon.points[1].y - y
        # rect = np.array([x, y, w, h], dtype=np.float32)
        
        if cv_img is None:
            return

        #! convert mask to label
        mask_img[mask_img > 0] = self.__label

        if self.__counter < 1000:
            text_file = open(str(self.__write_path + self.__obj_name + self.__text_filename), "a")
        
            p =  self.__img_path + str(self.__counter).zfill(8) + '.jpg'
            cv.imwrite(p, cv_img)
            text_file.write(p + " " + str(self.__label) + "\n")

            p =  self.__mask_path + str(self.__counter).zfill(8) + '.png'
            cv.imwrite(p, mask_img)
            text_file.write(p + " " + str(self.__label) + "\n")

            text_file.close()

        #cv.imshow("mask", mask_img)
        #cv.waitKey(3)
        
        print "writing counter: ", self.__counter
        self.__counter += 1

    def callback2(self, image_msg, mask_msg, rect_msg):
        cv_img = self.convert_image(image_msg)
        mask_img = self.convert_image(mask_msg, '8UC1')

        x = rect_msg.polygon.points[0].x
        y = rect_msg.polygon.points[0].y
        w = rect_msg.polygon.points[1].x - x
        h = rect_msg.polygon.points[1].y - y
        
        if cv_img is None:
            return

        #! convert mask to label
        mask_img[mask_img > 0] = self.__label

        if self.__counter < 500:
            text_file = open(str(self.__write_path + self.__obj_name + self.__text_filename), "a")
        
            im_p =  self.__img_path + str(self.__counter).zfill(8) + '.jpg'
            mk_p =  self.__mask_path + str(self.__counter).zfill(8) + '.png'
            rect_str = str(self.__label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
        
            text_file.write(im_p + ' ' + mk_p + ' ' + rect_str + "\n")

            cv.imwrite(im_p, cv_img)
            cv.imwrite(mk_p, mask_img)
        
            text_file.close()
        else:
            sys.exit()
        
        print "writing counter: ", self.__counter
        self.__counter += 1

    def subscribe(self):
        if self.__write_rect:
            image_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
            mask_sub = message_filters.Subscriber('/probability_map', Image)
            rect_sub = message_filters.Subscriber('/object_rect', Rect)
            ts = message_filters.TimeSynchronizer([image_sub, mask_sub, rect_sub], 20)
            ts.registerCallback(self.callback2)
        else:
            image_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
            mask_sub = message_filters.Subscriber('/probability_map', Image)
            ts = message_filters.TimeSynchronizer([image_sub, mask_sub], 20)
            ts.registerCallback(self.callback)

    def convert_image(self, image_msg, encoding = 'bgr8'):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, str(encoding))
        except Exception as e:
            print (e)
        return cv_img
        
def main(argv):
    try:
        rospy.init_node('image_rect_writer', anonymous = False)
        irw = ImageRectWriter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

if __name__ == "__main__":
    main(sys.argv)
