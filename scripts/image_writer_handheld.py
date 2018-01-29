#!/usr/bin/env python

import sys
import os
import shutil
import rospy
import rosbag
import json
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped as Rect
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

class ImageRectWriter(object):
    def __init__(self):
        self.__write_path = '/media/volume/Documents/datasets/hsr/'
        self.__text_filename = 'train.txt'
        self.__frame_write = 1000

        
        #! check the folder
        self.remove_empty_folders()

        #! auto label
        self.__label = len(os.walk(str(self.__write_path)).next()[1]) + 1
        rospy.loginfo('\033[34mLabel %s \033[0m', str(self.__label))

        numeric_label = rospy.get_param('~numeric_label', True)
        if numeric_label:
            self.__obj_name = str(self.__label).zfill(3) + '/'
        else:
            self.__obj_name = rospy.get_param('~object_name', None)


        ## hardcore
        self.__obj_name = "tea/"
            
        if self.__obj_name is None:
            rospy.logfatal('provide object name')
            sys.exit(1)

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

        self.__json_path = self.__write_path + self.__obj_name + 'json/'
        if not os.path.exists(str(self.__json_path)):
            os.makedirs(str(self.__json_path))
            
        self.subscribe()

    
    def remove_empty_folders(self):
        [
            shutil.rmtree(self.__write_path + fn)
            for fn in os.listdir(str(self.__write_path))
            if not os.path.isfile(self.__write_path + fn + '/' + self.__text_filename)
            if os.path.isdir(self.__write_path + fn)
        ]

    def callback(self, image_msg, mask_msg):
        cv_img = self.convert_image(image_msg)
        mask_img = self.convert_image(mask_msg, '8UC1')

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

        if self.__counter < self.__frame_write:
            text_file = open(str(self.__write_path + self.__obj_name + self.__text_filename), "a")
        
            im_p =  self.__img_path + str(self.__counter).zfill(8) + '.jpg'
            mk_p =  self.__mask_path + str(self.__counter).zfill(8) + '.png'
            rect_str = str(self.__label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
        
            text_file.write(im_p + ' ' + mk_p + ' ' + rect_str + "\n")

            ##! write in json
            image_fn = self.__obj_name + 'image/' + str(self.__counter).zfill(8) + '.jpg'
            mask_fn = self.__obj_name + 'mask/' + str(self.__counter).zfill(8) + '.png'
            json_label = self.json_encoding([x, y, w, h], image_fn, mask_fn)

            with open(self.__json_path + str(self.__counter).zfill(8) + '__labels.json') as f:
                json.dump(json_label, f)        
                
            cv.imwrite(im_p, cv_img)
            cv.imwrite(mk_p, mask_img)
        
            text_file.close()
        else:
            sys.exit()
        
        print "writing counter: ", self.__counter
        self.__counter += 1


    """
    encode to the ssd input format
    """
    def json_encoding(self, rect, image_fn, mask_fn):
        size = (float(rect[2]), float(rect[3]))
        cx, cy = (rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0)
        label = {}
        anno = {}
        anno['size'] = {'x' : size[0], 'y' : size[1]}
        anno['centre'] = {'x' : cx, 'y' : cy}
        anno['object_id'] = self.__label
        anno['label_type'] = 'box'
        anno['label_class'] = self.__obj_name
        label['labels'] = [anno]
        label['complete'] = 'null'
        label['image_filename'] = image_fn
        label['mask_filename'] = mask_fn
        return label
        
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
