#!/usr/bin/env python

import math
import random
import numpy as np
import cv2 as cv

class ArgumentationEngine(object):
    def __init__(self, im_width, im_height):
        self.__in_size = (im_width, im_height)
        self.__scales = np.array([1.5, 1.75, 2.0]) #! predefined scaling

    def process(self, in_data):

        im_rgb = in_data[:, :, 0:3].copy()
        im_dep = in_data[:, :, 3:6].copy()
        im_mask = in_data[:, :, 6:7].copy()
        
        flip_flag = random.randint(-1, 1)
        im_rgb = cv.flip(im_rgb, flip_flag)
        im_dep = cv.flip(im_dep, flip_flag)
        im_mask = cv.flip(im_mask, flip_flag)
        
        return self.generate_argumented_data(im_rgb, im_dep, im_mask)

    """
    Function to pack the template data and the search target region.
    """
    def generate_argumented_data(self, im_rgb, im_dep, im_mask):

        rect = self.bounding_rect(im_mask)

        ##! crop region around the object
        x, y, w, h = rect
        cx, cy = (x + w/2.0, y + h/2.0)

        sindx = int(random.randint(0, 2))
        s = self.__scales[sindx]

        nw = int(s * w)
        nh = int(s * h)
        nx = int(cx - nw/2.0)
        ny = int(cy - nh/2.0)

        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-im_rgb.shape[1]) if (nx+nw) > im_rgb.shape[1] else nw
        nh = nh-((ny+nh)-im_rgb.shape[0]) if (ny+nh) > im_rgb.shape[0] else nh

        bbox = np.array([nx, ny, nw, nh])
        
        rgb, dep, mask = self.crop_and_resize_inputs(im_rgb, im_dep, im_mask, bbox)

        r = random.randint(-min(w/2, h/2), min(w/2, h/2))
        box = bbox
        box[0] = bbox[0] + r
        box[1] = bbox[1] + r

        x, y, w, h = box            
        x2 = x + w
        y2 = y + h
        x = rect[0] if x > rect[0] else x
        y = rect[1] if y > rect[1] else y
        x = x + ((rect[0] + rect[2]) - x2) if x2 < rect[0] + rect[2] else x
        y = y + ((rect[1] + rect[3]) - y2) if y2 < rect[1] + rect[3] else y
        
        #! boarder conditions
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = w-(x2-im_rgb.shape[1]) if x2 > im_rgb.shape[1] else w
        h = h-(y2-im_rgb.shape[0]) if y2 > im_rgb.shape[0] else h

        box[0] = x
        box[1] = y
            
        rgb1, dep1, mask1 = self.crop_and_resize_inputs(im_rgb, im_dep, im_mask, box)
            
        templ_datum = self.pack_array(rgb, dep)
        tgt_datum = self.pack_array(rgb1, dep1, mask1)

        #! pack mask label in 4D
        target_datum = tgt_datum[0:6].copy()
        mask_datum = tgt_datum[6:7].copy()

        # cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
        # x,y,w,h = rect
        # cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
        # mask1 = mask_datum[0].copy()
        # mask1 = mask1.swapaxes(0, 1)
        # cv.namedWindow('img', cv.WINDOW_NORMAL)
        # cv.imshow('img', mask1)
        # cv.waitKey(0)

        return (templ_datum, target_datum, mask_datum)        

    def crop_and_resize_inputs(self, im_rgb, im_dep, im_mask, rect):
        x, y, w, h = rect
        rgb = im_rgb[y:y+h, x:x+w].copy()
        dep = im_dep[y:y+h, x:x+w].copy()
        msk = im_mask[y:y+h, x:x+w].copy()
        
        #! resize of network input
        rgb = cv.resize(rgb, (self.__in_size))
        dep = cv.resize(dep, (self.__in_size))
        msk = cv.resize(msk, (self.__in_size))            
        
        return rgb, dep, msk
        
    def pack_array(self, rgb, dep, mask = None):
        W = self.__in_size[1]
        H = self.__in_size[0]
        K = rgb.shape[2] + dep.shape[2]
        if not mask is None:
            K += 1
        datum = np.zeros((K, W, H), np.float)
        rgb = rgb.swapaxes(2, 0)
        rgb = rgb.swapaxes(2, 1)
        datum[0:3] = rgb
        
        dep = dep.swapaxes(2, 0)
        dep = dep.swapaxes(2, 1)
        datum[3:6] = dep

        if not mask is None:
            datum[6:7][0] = mask
        
        return datum


    def bounding_rect(self, im_mask):
        x1 = im_mask.shape[1] + 1
        y1 = im_mask.shape[0] + 1
        x2 = 0
        y2 = 0
        for j in xrange(0, im_mask.shape[0], 1):
            for i in xrange(0, im_mask.shape[0], 1):
                if im_mask[j, i] > 0:
                    x1 = i if i < x1 else x1
                    y1 = j if j < y1 else y1
                    x2 = i if i > x2 else x2
                    y2 = j if j > y2 else y2
        return np.array([x1, y1, x2 - x1, y2 - y1])
                    
    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(float)
        im_rgb[:, :, 0] -= float(104.0069879317889)
        im_rgb[:, :, 1] -= float(116.66876761696767)
        im_rgb[:, :, 2] -= float(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb
        
