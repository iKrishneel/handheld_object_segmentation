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
        
        if len(top) < 4:
            raise Exception('Three tops are required: template, target and label')

        try:
            params = eval(self.param_str)
            self.image_size_x = int(params['im_width'])
            self.image_size_y = int(params['im_height'])
            self.var_scale = bool(params.get('var_scale', False))
            self.random = bool(params.get('randomize', True))
            self.dataset_txt = str(params['filename'])
            self.directory = str(params['directory'])
            self.batch_size = int(params['batch_size'])
            
            if not os.path.isfile(self.dataset_txt):
                raise Exception('datasets textfile not found!')

            self.read_data_from_textfile2()
            # self.img_paths, self.dep_paths, self.lab_paths = self.read_data_from_textfile()

            # if self.img_paths.shape[0] != self.dep_paths.shape[0] != self.lab_paths.shape[0]:
            #     raise Exception('Empty text file')
            
            if self.random:
                random.seed()
                # self.idx = random.randint(0, len(self.img_paths)-1)
                
            self.__ae = ae.ArgumentationEngine(self.image_size_x, self.image_size_y, self.var_scale)

        except ValueError:
            raise ValueError('Parameter string missing or data type is wrong!')
            
    def reshape(self, bottom, top):
                
        n_images = self.batch_size
        out_size_x = int(self.image_size_x / 1)
        out_size_y = int(self.image_size_y / 1)
        
        top[0].reshape(n_images, 6, out_size_y, out_size_x) 
        top[1].reshape(n_images, 6, out_size_y, out_size_x) 
        top[2].reshape(n_images, 1, out_size_y, out_size_x)
        top[3].reshape(n_images, 1, 1, )
                
    def forward(self, bottom, top):

        for index in xrange(0, self.batch_size, 1):

            """
            indx = int(self.idx)
            im_rgb = cv.imread(self.img_paths[indx])
            im_dep = cv.imread(self.dep_paths[indx])
            im_mask = cv.imread(self.lab_paths[indx], cv.IMREAD_GRAYSCALE)

            template_datum, target_datum, label_datum = self.__ae.process2(im_rgb, im_dep, im_mask)
            
            ##! empty label (tmp test)
            if template_datum.shape == im_rgb.shape:
                while template_datum.shape == im_rgb.shape:
                    self.idx = random.randint(0, len(self.img_paths)-1)
                    indx = int(self.idx)
                    im_rgb = cv.imread(self.img_paths[indx])
                    im_dep = cv.imread(self.dep_paths[indx])
                    im_mask = cv.imread(self.lab_paths[indx], cv.IMREAD_GRAYSCALE)

                    template_datum, target_datum, label_datum = self.__ae.process2(im_rgb, im_dep, im_mask)

            """

            t_key, t_rnd, s_key, s_rnd = self.fetch_data_once()
            templ_data = self.__dataset[t_key][t_rnd]
            src_data = self.__dataset[s_key][s_rnd]
            
            im_trgb, im_tdep, im_tmask = self.read_images(**templ_data)
            im_srgb, im_sdep, im_smask = self.read_images(**src_data)

            while True:
                template_datum, _ = self.__ae.process2(im_trgb, im_tdep, im_tmask, True)
                if not template_datum is None:
                    break
                else:
                    t_key, t_rnd, _, _ = self.fetch_data_once()
                    templ_data = self.__dataset[t_key][t_rnd]
                    im_trgb, im_tdep, im_tmask = self.read_images(**templ_data)
            
            while True:
                target_datum, label_datum = self.__ae.process2(im_srgb, im_sdep, im_smask, False)
                if not target_datum is None:
                    break
                else:
                    _, _, s_key, s_rnd = self.fetch_data_once()
                    src_data = self.__dataset[s_key][s_rnd]
                    im_srgb, im_sdep, im_smask = self.read_images(**src_data)

            label = 1.0 if t_key == s_key else 0.0

            top[0].data[index] = template_datum.copy()
            top[1].data[index] = target_datum.copy()
            top[2].data[index] = label_datum.copy()
            top[3].data[index] = label
            
            # self.idx = random.randint(0, len(self.img_paths)-1)

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

        img_paths = []
        dep_paths = []
        lab_paths = []
        for line in img_lists:
            p = line.split(os.sep)
            dep_path = self.directory + p[-3] + '/depth/' + p[-1]
            msk_path = self.directory + p[-3] + '/mask/' + p[-1]
            img_paths.append(line)
            dep_paths.append(dep_path)
            lab_paths.append(msk_path)
            
        return np.array(img_paths), np.array(dep_paths), np.array(lab_paths)


    ######################################################################

    def read_textfile(self, filename):
        lines = [line.rstrip('\n')                                                                                              
                 for line in open(filename)                                                                             
        ]
        return np.array(lines)

    def read_images(self, **kwargs):
        im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
        im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
        im_mask = cv.imread(kwargs['mask'], cv.IMREAD_COLOR)
        return im_rgb, im_dep, im_mask

    def read_data_from_textfile2(self):
        train_fn = 'train.txt'
        self.__objects = self.read_textfile(os.path.join(self.directory, 'objects.txt'))

        self.__dataset = {}

        for obj in self.__objects:    
            fn = os.path.join(self.directory, os.path.join(obj, train_fn))
            if not os.path.isfile(fn):
                raise Exception('Missing data train.txt')
            lines = self.read_textfile(fn)
            datas = []
            for index, line in enumerate(lines):
                if index % 3 == 0:
                    datas.append({
                        'image': lines[index].split()[0],
                        'depth' : lines[index+1].split()[0],
                        'mask' :lines[index+2].split()[0]
                    })
            self.__dataset[str(obj)] = np.array(datas)


    def fetch_data_once(self):
        t_rnd = random.randint(0, self.__objects.shape[0] - 1)
        s_rnd = random.randint(0, self.__objects.shape[0] - 1)

        label = 1 if t_rnd is s_rnd else 0
    
        t_key = self.__objects[t_rnd]
        s_key = self.__objects[s_rnd]

        t_rnd = random.randint(0, (self.__dataset[t_key].shape[0]) - 1)
        s_rnd = t_rnd
    
        if label:
            s_rnd = t_rnd + 1 if t_rnd is 0 else t_rnd - 1
        else:
            s_rnd = random.randint(0, (self.__dataset[s_key].shape[0]) - 1)

        return (t_key, t_rnd, s_key, s_rnd)
