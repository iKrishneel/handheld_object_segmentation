#!/usr/bin/env python

import os
import sys
import cv2 as cv
import random
import numpy as np
from tqdm import tqdm

"""
Class for computing intersection over union(IOU)
"""
class JaccardCoeff:

    def iou(self, a, b):
        i = self.__intersection(a, b)
        if i == 0:
            return 0
        aub = self.__area(self.__union(a, b))
        anb = self.__area(i)
        area_ratio = self.__area(a)/self.__area(b)        
        score = anb/aub
        score /= area_ratio
        return score
        
    def __intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return 0
        else:
            return (x, y, w, h)
        
    def __union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def __area(self, rect):
        return np.float32(rect[2] * rect[3])

class SmoothMaskImage(object):
    def __init__(self, data_dir = None, objects_list = 'objects.txt', \
                 filename = 'train.txt'):
        if data_dir is None:
            raise ValueError('Provide dataset directory')
        if filename is None or not os.path.isfile(os.path.join(data_dir, filename)):
            raise ValueError('Provide image train.txt')
        if objects_list is None or not os.path.isfile(os.path.join(data_dir, objects_list)):
            raise ValueError('Provide image objects.txt')

        self.__dataset_dir = data_dir
        self.__objects = None
        self.__dataset = {}

        self.read_data_from_textfile(objects_list, filename)

        # self.smooth_image_edges()

        #! generate multiple instance
        self.__iou_thresh = 0.05
        self.__max_counter = 100

        im_bg = np.ones((480, 640, 3), np.uint8)
        
        im_bg = cv.imread("/home/krishneel/Desktop/image.jpg")
        self.argument(20, im_bg)
            

    def smooth_image_edges(self):

        cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
        for obj in self.__dataset:
            obj_data_list = self.__dataset[obj]
            for im_list in obj_data_list:
                [im_rgb, im_dep, im_mask] = self.read_images(**im_list)

                im_mask2 = im_mask.copy()
                im_mask, contours = self.edge_contour_points(im_mask)

                # file the gaps
                if not contours is None:
                    cv.drawContours(im_mask, contours, -1, (255, 255, 255), -1)

                rgb_mask = cv.bitwise_and(im_rgb, im_mask)
                if not contours is None:
                    cv.drawContours(rgb_mask, contours, -1, (0, 255, 0), 3)

                    # bounding rectangle
                    x, y, w, h = cv.boundingRect(contours[0])
                    cv.rectangle(rgb_mask, (x, y), (x+w, y+h), (255, 0, 255), 1)

                    #! save the json annotation file
                    
                # debug
                alpha = 0.3
                cv.addWeighted(im_rgb, alpha, rgb_mask, 1.0 - alpha, 0, rgb_mask)
                
                rgb_mask2 = cv.bitwise_and(im_rgb, im_mask2)
                z = np.hstack((rgb_mask, rgb_mask2))
                cv.imshow('image', z)
                cv.waitKey(20)
            

    @classmethod
    def edge_contour_points(self, im_mask):
        if im_mask is None:
            return im_mask, None

        #! smooth the edges in mask
        im_mask = cv.GaussianBlur(im_mask, (21, 21), 11.0)
        im_mask[im_mask > 150] = 255

        im_mask2 = im_mask.copy()
        if len(im_mask2.shape) == 3:
            im_mask2 = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)

        _, im_mask2 = cv.threshold(im_mask2, 127, 255, 0)
        _ , contours, _ = cv.findContours(im_mask2, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

        #! remove noisy contours
        max_area = 0
        contour_obj = None
        
        for contour in contours:
            area = cv.contourArea(contour)
            max_area, contour_obj = (area, contour) if area > max_area \
                                    else (max_area, contour_obj)

        if max_area < 16 ** 2:
            return im_mask, None
        else:
            return im_mask, [contour_obj]

            
    @classmethod
    def read_textfile(self, filename):
        lines = [line.rstrip('\n')
                 for line in open(filename)                                                                             
        ]
        return np.array(lines)

    @classmethod
    def read_images(self, **kwargs):
        im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
        im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
        im_mask = cv.imread(kwargs['mask'], cv.IMREAD_COLOR)
        return [im_rgb, im_dep, im_mask]
    
    def read_data_from_textfile(self, objects_list, filename):
        self.__objects = self.read_textfile(os.path.join(self.__dataset_dir, objects_list))
        for obj in self.__objects:
            fn = os.path.join(self.__dataset_dir, os.path.join(obj, filename))
            if not os.path.isfile(fn):
                raise Exception('Missing data train.txt')
            lines = self.read_textfile(fn)
            datas = []
            #for index, line in enumerate(lines):
            for index, line in enumerate(tqdm(lines, ascii = True)):
                if index % 3 == 0:
                    datas.append({
                        'image': lines[index].split()[0],
                        'depth' : lines[index+1].split()[0],
                        'mask' :lines[index+2].split()[0]
                    })
            self.__dataset[str(obj)] = np.array(datas)

            
    def argument(self, num_proposals, im_bg, im_mk = None, mrect = None):
        im_y, im_x, _ = im_bg.shape
        flag_position = []
        img_output = im_bg.copy()

        mask_output = np.zeros((im_y, im_x, 1), np.uint8)
        if not im_mk is None:
            mask_output = im_mk.copy()
        if not mrect is None:
            flag_position.append(mrect)

        labels = []
        for index in xrange(0, num_proposals, 1):

            while(True):
                ##! randomly select object
                label = random.randint(0, len(self.__dataset) - 1)
                
                ##! randomly select item
                idx = random.randint(0, len(self.__dataset.items()[label][1]))
            
                im_list =  self.__dataset.items()[label][1][idx]
                [image, depth, mask] = self.read_images(**im_list)

                mask, contours = self.edge_contour_points(mask)
                if not contours is None:
                    rect = cv.boundingRect(contours[0])
                    x,y,w,h = rect
                    # cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 1)            
                    break
            
            
                
            # idx = random.randint(0, len(self.__img_paths)-1)
            # im_path = self.__img_paths[idx]
            # mk_path = self.__mask_paths[idx]
            # label = self.__labels[idx]
            # rect = self.__rects[idx]
            # x,y,w,h = rect

            # image = cv.imread(im_path)
            # mask = cv.imread(mk_path)
            
            # mask[mask > 0] = 255

            flip_flag = random.randint(-1, 2)
            #### TEMP
            flip_flag = 2
            if flip_flag > -2 and flip_flag < 2:
                rect_copy = rect.copy()
                image, rect = self.flip_image(image, [rect], flip_flag)
                mask, _ = self.flip_image(mask, [rect_copy], flip_flag)
                x,y,w,h = rect[0]
            
            im_roi = image[y:y+h, x:x+w].copy()
            im_msk = mask[y:y+h, x:x+w].copy()
            
            resize_flag = random.randint(0, 1)
            if resize_flag:
                scale = random.uniform(1.0, 2.2)
                w = int(w * scale)
                h = int(h * scale)
                im_roi = cv.resize(im_roi, (int(w), int(h)))
                im_msk = cv.resize(im_msk, (int(w), int(h)))
                rect  = np.array([x, y, w, h], dtype=np.int)
            
            cx, cy = random.randint(0, im_x - 1), random.randint(0, im_y-1)
            cx = cx - ((cx + w) - im_x) if cx + w > im_x - 1 else cx
            cy = cy - ((cy + h) - im_y) if cy + h > im_y - 1 else cy
            nrect = np.array([cx, cy, w, h])

            ##! and-ing to remove bg pixels
            im_msk2 = im_msk.copy()
            im_msk[im_msk < 127] = 0
            im_roi = cv.bitwise_and(im_roi, im_msk)
            
            z = np.hstack((im_msk, im_roi))
            cv.imshow('imask', z)
    
            
            counter = 0
            position_found = True
            if len(flag_position) > 0:
                jc = JaccardCoeff()
                for bbox in flag_position:
                    if jc.iou(bbox, nrect) > self.__iou_thresh and position_found:
                        is_ok = True
                        while True:
                            cx, cy = random.randint(0, im_x - 1), random.randint(0, im_y-1)
                            cx = cx - ((cx + w) - im_x) if cx + w > im_x - 1 else cx
                            cy = cy - ((cy + h) - im_y) if cy + h > im_y - 1 else cy
                            nrect = np.array([cx, cy, w, h])
                            for bbox2 in flag_position:
                                if jc.iou(bbox2, nrect) > self.__iou_thresh:
                                    is_ok = False
                                    break
                            if is_ok:
                                break

                            counter += 1
                            if counter > self.__max_counter:
                                position_found = False
                                break
            if position_found:
                im_roi = cv.bitwise_and(im_roi, im_msk)
                for j in xrange(0, h, 1):
                    for i in xrange(0, w, 1):
                        nx, ny = i + cx, j + cy
                        if im_msk[j, i, 0] > 0 and nx < im_x and ny < im_y:
                            img_output[ny, nx] = im_roi[j, i]
                            mask_output[ny, nx] = label + 1 ##! check this
                
                
                """
                print im_msk.shape, mask_output.shape
                # img_output[cy:cy+h, cx:cx+w, :] = im_roi
                center = (cx + w/2, cy + h/2)
                im_msk[im_msk>=0] = 255
                img_output = cv.seamlessClone(im_roi, img_output, im_msk, center, cv.NORMAL_CLONE)
                
                im_msk[im_msk > 127] = label + 1
                mask_output[cy:cy+h, cx:cx+w, 0] = im_msk[:, :, 1]
                # mask_output[mask_output <= 127] = 0
                # mask_output[mask_output > 127] = label + 1 # plus background
                """ 
               
                flag_position.append(nrect)
                labels.append(label)

        ###! debug
        debug = True
        if debug:
            for r in flag_position:
                x,y,w,h = r
                cv.rectangle(img_output, (x,y), (x+w, h+y), (0, 255, 0), 3)
                cv.namedWindow('roi', cv.WINDOW_NORMAL)
                cv.imshow('roi', img_output)

            im_flt = mask_output.astype(np.float32)
            # im_flt = cv.normalize(im_flt, 0, 1, cv.NORM_MINMAX)
            im_flt /= len(self.__dataset)
            im_flt *= 255.0
            im_flt = im_flt.astype(np.uint8)
            im_flt = cv.applyColorMap(im_flt, cv.COLORMAP_JET)
            
            cv.imshow('mask2', im_flt)
            mask_output *= 255
            cv.imshow('mask', mask_output)
            cv.waitKey(0)
        ###! end-debug
                
        return (img_output, mask_output, np.array(flag_position), np.array(labels))



def main(argv):
    if len(argv) < 2:
        raise ValueError('Provide image list.txt')

    smi = SmoothMaskImage(argv[1], argv[2], argv[3])
    

if __name__ == '__main__':
    main(sys.argv)
