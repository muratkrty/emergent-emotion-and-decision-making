# Author: Murat Kirtay, The BioRobotics Inst./SSSA/
# Date: 22/06/2016
# Description: Preprocess images for EE-DM project

import cv2
import os
import re 
import numpy as np
import random as rd

def print_motto():
    print "... one must build a thing to truly understand it."

def binarize_patterns(ppath, dpath):
    ''' Create binary version of the patterns and save it to  
        a predefined directory. Note: background is white.
        TODO: check treshold func parameters
    '''

    for dname, sdir, fnames in os.walk(ppath):
        for pname in fnames:
            #img = cv2.imread(os.path.join(ppath, pname))
            img = os.path.join(ppath, pname)
            graypt = cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            bin_img = cv2.threshold(graypt, 20, 255, cv2.THRESH_BINARY)[1]
            name = str('bin_' +pname)

            cv2.imwrite(os.path.join(dpath, name), bin_img)
            
def resize_patterns(ppath, rpath, resize_wh):
    ''' Resize a given pattern (to default20x20) and save it to 
        a predefined directory
    '''

    width, height = resize_wh[0], resize_wh[1]

    for dname, sdir, fnames in os.walk(ppath):
        for pname in fnames:
            npname = os.path.join(ppath, pname)
            npimg = cv2.imread(npname,0)
            resized_pattern = cv2.resize(npimg, resize_wh)

            cv2.imwrite(os.path.join(rpath, pname), resized_pattern)
    
def create_noisy_patterns(patterns, noisyp, rate):
    ''' Walk through trained patterns folder and create
        noisy pattern based on given contamination rate
    '''

    white, black= 255, 0

    for dname, sdir, fnames in os.walk(patterns):
        for pname in fnames:
            noisyp_name = os.path.join(patterns, pname)
            img = cv2.imread(noisyp_name,0)
            noise_rate =  int(img.size * rate)
            w, h = img.shape
            for i in range(noise_rate):
                pixel = img[rd.randint(1, h-1)][rd.randint(1, w-1)]
                if pixel > 0:
                    img[rd.randint(1, h-1)][rd.randint(1, w-1)] = black
                else:
                    img[rd.randint(1, h-1)][rd.randint(1, w-1)] = white
            cv2.imwrite(os.path.join(noisyp, pname), img)


