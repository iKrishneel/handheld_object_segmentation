#!/usr/bin/env python

import sys
import os
import math
import random
import shutil
import caffe
import lmdb
import numpy as np
import cv2 as cv

(major, minor, _) = cv.__version__.split(".")


class ProcessRGBImage:
    def __init__(self):

        self.__in_size = (224, 224)
        
        self.__directory = '/home/krishneel/Documents/datasets/handheld_objects/'
        self.__textfile = self.__directory + 'train.txt'
        
        if not os.path.isfile(self.__textfile):
            print 'ERROR: train.txt file not found'
            sys.exit()

        lmdb_folder = self.__directory + 'lmdb/'
        if not os.path.exists(lmdb_folder):
            os.makedirs(lmdb_folder)
            
        self.__lmdb_images = lmdb_folder + 'features'
        if os.path.exists(self.__lmdb_images):
            shutil.rmtree(self.__lmdb_images)
            
        self.process()

    def process(self):
        lines = self.read_textfile(self.__textfile)
        #np.random.shuffle(lines)

        # setup
        map_size = 1e12
        #lmdb_images = lmdb.open(str(self.__lmdb_images), map_size=int(map_size))
        #with lmdb_images.begin(write=True) as img_db:
        for z in xrange(0, 1, 1):
            for index in xrange(0, len(lines), 3):
                line1 = lines[index].split()[0]
                line2 = lines[index+1].split()[0]
                line3 = lines[index+2].split()[0]
                im_rgb = cv.imread(str(line1))
                im_dep = cv.imread(str(line2))
                im_mask = cv.imread(str(line3), 0)

                data_array = self.create_training_data(im_rgb, im_dep, im_mask)

                W = self.__in_size[0]
                H = self.__in_size[1]
                K = im_dep.shape[2] + im_rgb.shape[2] + 1
                
                for j in xrange(0, len(data_array), 3):
                    ##! write to lmdb
                    rgb = data_array[j].copy()
                    dep = data_array[j+1].copy()
                    msk = data_array[j+2].copy()

                    ##! change to network input size
                    rgb = cv.resize(rgb, (self.__in_size))
                    dep = cv.resize(dep, (self.__in_size))
                    msk = cv.resize(msk, (self.__in_size))

                    datum = np.zeros((K, W, H), np.float)

                    rgb = rgb.swapaxes(2, 0)
                    rgb = rgb.swapaxes(2, 1)
                    datum[0:3] = rgb
                
                    dep = dep.swapaxes(2, 0)
                    dep = dep.swapaxes(2, 1)
                    datum[3:6] = dep

                    datum[6:7][0] = msk

                    #cv.imshow("dep", data_array[j+1])
                    #cv.imshow("mask", data_array[j+2])
                    #cv.imshow("rgb", data_array[j])
                    #cv.waitKey(0)

        #lmdb_images.close()


    def create_training_data(self, im_rgb, im_dep, im_mask, num_samples = 3):

        mask, rect = self.create_mask_labels(im_mask)
        if rect is None:
            return

        ##! crop region around the object
        x, y, w, h = rect
        cx, cy = (x + w/2.0, y + h/2.0)
        
        scales = np.array([1.5, 1.75, 2.0])

        rects = []
        data_array = []
        for s in scales:
            nw = int(s * w)
            nh = int(s * h)
            nx = int(cx - nw/2.0)
            ny = int(cy - nh/2.0)

            nx = 0 if nx < 0 else nx
            ny = 0 if ny < 0 else ny
            nw = nw-((nx+nw)-im_rgb.shape[1]) if (nx+nw) > im_rgb.shape[1] else nw
            nh = nh-((ny+nh)-im_rgb.shape[0]) if (ny+nh) > im_rgb.shape[0] else nh

            rects.append([nx, ny, nw, nh])
            
            rgb = im_rgb[ny:ny+nh, nx:nx+nw].copy()
            dep = im_dep[ny:ny+nh, nx:nx+nw].copy()
            msk = mask[ny:ny+nh, nx:nx+nw].copy()

            ##! normalize data
            rgb = self.demean_rgb_image(rgb)
            dep = dep.astype(np.float)
            dep /= dep.max()

            data_array.append(rgb)
            data_array.append(dep)
            data_array.append(msk)

        for im, bb in zip(data_array, rects):
            #for i in xrange(0, num_samples, 1):
            r = random.randint(-min(w/2, h/2), min(w/2, h/2))
            print bb
            box = bb
            box[0] = bb[0] + r
            box[1] = bb[1] + r

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

            

            print box, " ", rect
            print (x2 - (rect[0] + rect[2]))

            cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)

            x,y,w,h = rect
            cv.rectangle(im_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)

            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', im_rgb)
            cv.waitKey(0)
            
        return data_array

        
        
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
        if major < 3:
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

        mask = np.asarray(im_gray, np.float_)
        mask = mask / mask.max()

        rect = cv.boundingRect(contour[index]) if index > -1 else None
        
        return (mask, rect)


    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(float)
        im_rgb[:, :, 0] -= float(104.0069879317889)
        im_rgb[:, :, 1] -= float(116.66876761696767)
        im_rgb[:, :, 2] -= float(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb
        
    def read_textfile(self, path_to_txt):
        lines = [line.rstrip('\n')
                 for line in open(path_to_txt)
        ]
        return lines

def main(argv):
    pi = ProcessRGBImage()
    
if __name__ == '__main__':
    main(sys.argv)
