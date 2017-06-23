#!/usr/bin/env python

import os
import random
import caffe
import numpy as np
import cv2 as cv
import argumentation_engine as ae

class DataArgumentationLayer(caffe.Layer):

    def setup(self, bottom, top):
        #! check that inputs are pair of image and labels (bbox and label:int)
        if len(bottom) != 0:
            raise Exception('This layer does not take any bottom')
        
        if len(top) < 3:
            raise Exception('Three tops are required: template, target and label')

        try:
            params = eval(self.param_str)
            self.image_size_x = int(params['im_width'])
            self.image_size_y = int(params['im_height'])
            #self.mean = np.array(params['mean'])
            self.random = bool(params.get('randomize', True))
            self.dataset_txt = str(params['filename'])
            self.directory = str(params['directory'])
            self.batch_size = int(params['batch_size'])
            
            if not os.path.isfile(self.dataset_txt):
                raise Exception('dataset textfile not found!')

            self.lines = self.read_data_from_textfile()
            if len(self.lines) < 3:
                raise Exception('Empty text file')
            
            if self.random:
                random.seed()
                self.idx = random.randint(0, (len(self.lines)/3)-1)
                
            self.__ae = ae.ArgumentationEngine(self.image_size_x, self.image_size_y)

        except ValueError:
            raise ValueError('Parameter string missing or data type is wrong!')
            
    def reshape(self, bottom, top):
                
        n_images = self.batch_size
        out_size_x = int(self.image_size_x / 1)
        out_size_y = int(self.image_size_y / 1)
        
        top[0].reshape(n_images, 6, out_size_y, out_size_x) 
        top[1].reshape(n_images, 6, out_size_y, out_size_x) 
        top[2].reshape(n_images, 1, out_size_y, out_size_x)
                
    def forward(self, bottom, top):

        for index in xrange(0, self.batch_size, 1):
            indx = int(self.idx * 3)
            im_rgb = cv.imread(self.lines[indx])
            im_dep = cv.imread(self.lines[indx+1])
            im_mask = cv.imread(self.lines[indx+2], 0)
            
            template_datum, target_datum, label_datum = self.__ae.process2(im_rgb, im_dep, im_mask)

            top[0].data[index] = template_datum.copy()
            top[1].data[index] = target_datum.copy()
            top[2].data[index] = label_datum.copy()

            self.idx = random.randint(0, (len(self.lines)/3)-1)

    def backward(self, top, propagate_down, bottom):
        pass

        
    def read_data_from_textfile(self):
        lines = [line.rstrip('\n')
                 for line in open(self.dataset_txt)
        ]

        #! its img, depth, mask
        img_lists = []
        for index in xrange(0, len(lines), 3):
            img_lists.append(lines[index].split()[0])

        lines = []
        for line in img_lists:
            p = line.split(os.sep)
            dep_path = self.directory + p[-3] + '/depth/' + p[-1]
            msk_path = self.directory + p[-3] + '/mask/' + p[-1]
            lines.append(line)
            lines.append(dep_path)
            lines.append(msk_path)

        return np.array(lines)
