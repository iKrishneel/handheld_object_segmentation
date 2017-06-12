#!/usr/bin/env python

import caffe
import argumentation_engine as ae

class DataArgumentationLayer(caffe.Layer):

    def setup(self, bottom, top):
        #! check that inputs are pair of image and labels (bbox and label:int)
        if len(bottom) != 1:
            raise Exception('Need two inputs for Argumentation')
        
        if len(top) < 3:
            raise Exception('Current Implementation needs 6 top blobs')

        try:
            plist = self.param_str.split(',')
            self.image_size_x = int(plist[0])
            self.image_size_y = int(plist[1])
            self.__ae = ae.ArgumentationEngine(self.image_size_x, self.image_size_y)

        except ValueError:
            raise ValueError('Parameter string missing or data type is wrong!')

            
    def reshape(self, bottom, top):
        #if bottom[0].data < 5:
        #    raise Exception('Labels should be 5 dimensional vector')
                
        n_images = bottom[0].data.shape[0]        
        out_size_x = int(self.image_size_x / 1)
        out_size_y = int(self.image_size_y / 1)
        
        top[0].reshape(n_images, 6, out_size_y, out_size_x) 
        top[1].reshape(n_images, 6, out_size_y, out_size_x) 
        top[2].reshape(n_images, 1, out_size_y, out_size_x)
                
    def forward(self, bottom, top):
        for index, data in enumerate(bottom[0].data):
            
            template_datum, target_datum, label_datum = self.__ae.process(data.copy())
            
            top[0].data[index] = template_datum.copy()
            top[1].data[index] = target_datum.copy()
            top[2].data[index] = label_datum.copy()

    def backward(self, top, propagate_down, bottom):
        pass
        

