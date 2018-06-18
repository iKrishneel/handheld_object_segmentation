#!/usr/bin/env python

import sys
import os
import shutil
import json
import numpy as np
import cv2 as cv

class ImageAnnotationWriter(object):
    def __init__(self, write_dir, num_frames_write = 20, numeric_label = None):

        if not os.path.isdir(write_dir):
            print ('write_dir is not a valid directory %s' %write_dir)
            sys.exit()
        
        self.__write_path = write_dir
        
        self.__text_filename = 'train.txt'
        self.__frame_write = num_frames_write
        
        #! check the folder
        self.remove_empty_folders()

        #! auto label
        self.__label = len(os.walk(str(self.__write_path)).next()[1]) + 1
        print('\033[34mLabel %s \033[0m'% str(self.__label))
            
        if numeric_label:
            self.__obj_name = str(self.__label).zfill(3)

        if self.__obj_name is None:
            print('provide object name')
            sys.exit(1)

        self.__counter = 0

        self.__write_depth = True
        
        if not os.path.exists(os.path.join(self.__write_path, self.__obj_name)):
            os.makedirs(os.path.join(self.__write_path, self.__obj_name))

        self.__anno_path = os.path.join(self.__write_path, os.path.join(self.__obj_name, 'annotations'))
        self.__img_path = os.path.join(self.__write_path, os.path.join(self.__obj_name, 'image'))
        self.__mask_path = os.path.join(self.__write_path, os.path.join(self.__obj_name, 'mask'))
        
        self.make_directory(self.__img_path)
        self.make_directory(self.__mask_path)
        self.make_directory(self.__anno_path)

        if self.__write_depth:
            self.__dep_path = os.path.join(self.__write_path, os.path.join(self.__obj_name, 'depth'))
            self.make_directory(self.__dep_path)

        print ('Write Directory Info')
        print ('\t-> %s' % self.__img_path)
        print ('\t-> %s' % self.__dep_path)
        print ('\t-> %s' % self.__mask_path)
                
        object_list = open(os.path.join(self.__write_path, 'objects.txt'), 'a')
        object_list.write(str(self.__obj_name) + "\n")
        object_list.close()

    def make_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def remove_empty_folders(self):
        [
            shutil.rmtree(os.path.join(self.__write_path, fn))
            for fn in os.listdir(str(self.__write_path))
            if not os.path.isfile(os.path.join(self.__write_path, os.path.join(fn, self.__text_filename)))
            if os.path.isdir(os.path.join(self.__write_path, fn))
        ]

    def callback2(self, image_msg, mask_msg, depth_msg):
        cv_img = self.convert_image(image_msg)
        mask_img = self.convert_image(mask_msg, '8UC1')
        depth_img = sef.convert_image(depth_msg, '32FC1')
        
        if cv_img is None or mask_img is None or depth_img is None:
            return

        self.write_data_to_memory(cv_img, mask_img, depth_img)

    def write_data_to_memory(self, cv_img, mask_img, depth_img, bbox = None):
        
        #! convert mask to label
        mask_img[mask_img > 0] = 255

        if self.__counter < self.__frame_write:
            im_p =  os.path.join(self.__img_path, str(self.__counter).zfill(8) + '.jpg')
            cv.imwrite(im_p, cv_img)
            
            mk_p =  os.path.join(self.__mask_path, str(self.__counter).zfill(8) + '.jpg')
            cv.imwrite(mk_p, mask_img)
            
            if self.__write_depth:
                dp_p =  os.path.join(self.__dep_path, str(self.__counter).zfill(8) + '.jpg')
                cv.imwrite(dp_p, depth_img)

            im_p = im_p.replace(self.__write_path, '')
            mk_p = mk_p.replace(self.__write_path, '')
            dp_p = dp_p.replace(self.__write_path, '')
            
            # text_file.write(im_p + ' ' + mk_p + ' ' + dp_p + "\n")

            anno = {
                'image': im_p,
                'depth': dp_p,
                'mask': mk_p,
                'label': self.__label
            }

            if not bbox is None:
                anno['bbox'] = list(bbox)

            json_fn = os.path.join(self.__anno_path, str(self.__counter).zfill(8) + '.json')
            with open(json_fn, 'w') as f:
                json.dump(anno, f)        

            text_file = open(os.path.join(self.__write_path,
                                          os.path.join(self.__obj_name, self.__text_filename)), 'a')
            text_file.write(json_fn + "\n")
            text_file.close()

            print "writing counter: ", self.__counter
            self.__counter += 1
        else:
            print ("Required number of frames written. Writing stopped...")
            
            # sys.exit()
        

