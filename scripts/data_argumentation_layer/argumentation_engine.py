#!/usr/bin/env python

import math
import random
import numpy as np
import cv2 as cv

(CV_MAJOR, CV_MINOR, _) = cv.__version__.split(".")

class ArgumentationEngine(object):
    def __init__(self, im_width, im_height, var_scaling = False):
        self.__in_size = (im_width, im_height)
        self.__scales = np.array([1.25, 1.5, 1.75, 2.0, 2.25, 2.5]) #! predefined scaling
        self.__variable_scaling = var_scaling

    def process(self, in_data):

        im_rgb = in_data[:, :, 0:3].copy()
        im_dep = in_data[:, :, 3:6].copy()
        im_mask = in_data[:, :, 6:7].copy()
        
        flip_flag = random.randint(-1, 1)
        im_rgb = cv.flip(im_rgb, flip_flag)
        im_dep = cv.flip(im_dep, flip_flag)
        im_mask = cv.flip(im_mask, flip_flag)
        
        return self.generate_argumented_data(im_rgb, im_dep, im_mask)

    def process2(self, in_rgb, in_dep, in_mask, is_templ = True):
        flip_flag = random.randint(-1, 1)
        im_rgb = cv.flip(in_rgb, flip_flag)
        im_dep = cv.flip(in_dep, flip_flag)
        im_mask = cv.flip(in_mask, flip_flag)

        if len(im_mask.shape) == 3:
            im_mask = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)

        im_rgb = self.demean_rgb_image(im_rgb)
        im_dep = im_dep.astype(np.float32)
        # im_dep /= im_dep.max()  ## >> no global only local normalization for depth
        
        return self.generate_argumented_data(im_rgb, im_dep, im_mask, is_templ)

    """
    Function to pack the template data and the search target region.
    """
    def generate_argumented_data(self, im_rgb, im_dep, in_mask, is_templ = True):


        # rect = self.bounding_rect(im_mask)
        im_mask, rect = self.create_mask_labels(in_mask)

        if rect is None:
            return None, None
        
        x, y, w, h = rect
        
        if rect is None or im_mask is None:
            return im_rgb, im_dep, im_mask

        im_mask = im_mask.astype(np.float32)
        im_mask /= im_mask.max()
        
        if is_templ:
            ##! crop region around the object

            sindx = int(random.randint(0, len(self.__scales) - 1))
            s = self.__scales[sindx]
            bbox = self.get_region_bbox(im_rgb, rect, s)
            rgb, dep, mask = self.crop_and_resize_inputs(im_rgb, im_dep, im_mask, bbox, True)
            
            templ_datum = self.pack_array(rgb, dep)
            return templ_datum, _
        
        sindx = int(random.randint(0, len(self.__scales) - 1))
        s = self.__scales[sindx]
        bbox = self.get_region_bbox(im_rgb, rect, s)
        ###### target cropping
        if self.__variable_scaling:
            sindx = int(random.randint(0, len(self.__scales) - 1))
            s = self.__scales[sindx]
            bbox = self.get_region_bbox(im_rgb, rect, s)

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
            
        rgb1, dep1, mask1 = self.crop_and_resize_inputs(im_rgb, im_dep, im_mask, box, True)

        #####
        ## ToDo: color space argumentation
        ####

            

        tgt_datum = self.pack_array(rgb1, dep1, mask1)

        #! pack mask label in 4D
        target_datum = tgt_datum[0:6].copy()
        mask_datum = tgt_datum[6:7].copy()
        mask_datum[mask_datum > 0.0] = 1.0
        
        #! for softmaxloss
        mask_datum = mask_datum.astype(np.uint8)


        
        ##################################
        # cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
        # x,y,w,h = rect
        # cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
        # mask1 = mask_datum[0].copy()
        # mask1 = mask1.swapaxes(0, 1)
        
        # z = np.hstack((rgb1, dep1))
        # cv.namedWindow('img', cv.WINDOW_NORMAL)
        # cv.imshow('img', z)
        # cv.waitKey(3 )
        ##################################
        
        return (target_datum, mask_datum)        

    def get_region_bbox(self, im_rgb, rect, s):
        x, y, w, h = rect
        cx, cy = (x + w/2.0, y + h/2.0)

        sindx = int(random.randint(0, len(self.__scales) - 1))
        s = self.__scales[sindx]

        nw = int(s * w)
        nh = int(s * h)
        nx = int(cx - nw/2.0)
        ny = int(cy - nh/2.0)

        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-im_rgb.shape[1]) if (nx+nw) > im_rgb.shape[1] else nw
        nh = nh-((ny+nh)-im_rgb.shape[0]) if (ny+nh) > im_rgb.shape[0] else nh

        return np.array([nx, ny, nw, nh])

    def crop_and_resize_inputs(self, im_rgb, im_dep, im_mask, rect, is_norm_dep = True):
        x, y, w, h = rect
        rgb = im_rgb[y:y+h, x:x+w].copy()
        dep = im_dep[y:y+h, x:x+w].copy()
        msk = im_mask[y:y+h, x:x+w].copy()

        if is_norm_dep:
            dep /= dep.max()
        
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
        datum = np.zeros((K, W, H), np.float32)
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
        im_rgb = im_rgb.astype(np.float32)
        im_rgb[:, :, 0] -= np.float32(104.0069879317889)
        im_rgb[:, :, 1] -= np.float32(116.66876761696767)
        im_rgb[:, :, 2] -= np.float32(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb
        
    def create_mask_labels(self, im_mask):
        if len(im_mask.shape) is None:
            print 'ERROR: Empty input mask'
            return

        thresh_min = 100
        thresh_max = 255
        im_gray = im_mask.copy()
        im_gray[im_gray > thresh_min] = 255
        im_gray[im_gray <= thresh_min] = 0

        ##! fill the gap
        if CV_MAJOR < str(3):
            contour, hier = cv.findContours(im_gray.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        else:
            im, contour, hier = cv.findContours(im_gray.copy(), cv.RETR_CCOMP, \
                                                cv.CHAIN_APPROX_SIMPLE)

        max_area = 0
        index = -1
        for i, cnt in enumerate(contour):
            cv.drawContours(im_gray, [cnt], 0, 255, -1)

            a = cv.contourArea(cnt)
            if max_area < a:
                max_area = a
                index = i

        mask = None
        if index > -1:
            mask = np.asarray(im_gray, np.float32)
            mask = mask / mask.max()

        rect = cv.boundingRect(contour[index]) if index > -1 else None

        return (mask, rect)




# direct = '/home/krishneel/Documents/datasets/handheld_objects/'
# # im_rgb = cv.imread(direct + 'image/00000000.jpg')
# # im_dep = cv.imread(direct + 'depth/00000000.jpg')
# # im_mask = cv.imread(direct + 'mask/00000000.jpg')

# # print im_rgb.shape, im_dep.shape, im_mask.shape

# # ae = ArgumentationEngine(227, 227)
# # ae.process2(im_rgb, im_dep, im_mask)

# import os

# def read_textfile(filename):
#     lines = [line.rstrip('\n')                                                                                              
#              for line in open(filename)                                                                             
#     ]
#     return np.array(lines)


# def read_images(**kwargs):
#     im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
#     im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
#     im_mask = cv.imread(kwargs['mask'], cv.IMREAD_COLOR)
#     return im_rgb, im_dep, im_mask

# train_fn = 'train.txt'
# objects = read_textfile(os.path.join(direct, 'objects.txt'))

# dataset = {}

# for obj in objects:    
#     fn = os.path.join(direct, os.path.join(obj, train_fn))
#     if not os.path.isfile(fn):
#         raise Exception('Missing dataset train.txt')
    
#     lines = read_textfile(fn)

#     datas = []
#     for index, line in enumerate(lines):
#         if index % 3 == 0:
#             datas.append({
#                 'image': lines[index].split()[0],
#                 'depth' : lines[index+1].split()[0],
#                 'mask' :lines[index+2].split()[0]
#             })

#     dataset[str(obj)] = np.array(datas)


# def fetch_data_once():
#     t_rnd = random.randint(0, objects.shape[0] - 1)
#     s_rnd = random.randint(0, objects.shape[0] - 1)

#     label = 1 if t_rnd is s_rnd else 0
    
#     t_key = objects[t_rnd]
#     s_key = objects[s_rnd]

#     t_rnd = random.randint(0, (dataset[t_key].shape[0]) - 1)
#     s_rnd = t_rnd
    
#     if label:
#         s_rnd = t_rnd + 1 if t_rnd is 0 else t_rnd - 1
#     else:
#         s_rnd = random.randint(0, (dataset[s_key].shape[0]) - 1)

#     return (t_key, t_rnd, s_key, s_rnd)
    

# ae = ArgumentationEngine(227, 227)

# for i in xrange(0, 100, 1):
#     """
#     t_rnd = random.randint(0, objects.shape[0] - 1)
#     s_rnd = random.randint(0, objects.shape[0] - 1)

#     label = 1 if t_rnd is s_rnd else 0
    
#     t_key = objects[t_rnd]
#     s_key = objects[s_rnd]

#     t_rnd = random.randint(0, (dataset[t_key].shape[0]) - 1)
#     s_rnd = t_rnd
    
#     if label:
#         s_rnd = t_rnd + 1 if t_rnd is 0 else t_rnd - 1
#     else:
#         s_rnd = random.randint(0, (dataset[s_key].shape[0]) - 1)
#     """


    
#     t_key, t_rnd, s_key, s_rnd = fetch_data_once()
    
#     templ_data = dataset[t_key][t_rnd]
#     src_data = dataset[s_key][s_rnd]

#     im_trgb, im_tdep, im_tmask = read_images(**templ_data)
#     im_srgb, im_sdep, im_smask = read_images(**src_data)

#     # ae.process2(im_trgb, im_tdep, im_tmask)

#     while True:
#         src_datum, mask_datum = ae.process2(im_srgb, im_sdep, im_smask, False)
#         if not src_datum is None:
#             break
#         else:
#             print "None Detected"
#             _, _, s_key, s_rnd = fetch_data_once()
#             src_data = dataset[s_key][s_rnd]
#             im_srgb, im_sdep, im_smask = read_images(**src_data)
        
#     print "DONE:", src_datum.shape
