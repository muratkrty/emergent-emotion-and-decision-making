# Author: Murat Kirtay, The BioRobotics Inst./SSSA/
# Date: 22/06/2016
# Description: Preprocess images for EE-DM project
#              Contaminate a images with predefined contamination rate
# Notes: 0- Final pattern image size should be 20x20

import image_processing as imp


def main():

    training = 'patterns/training'
    resized = 'patterns/resized'
    binarized = 'patterns/binarized'

    # d folders are used for displa purpose
    dfolders = {0 :'dpatterns/noise/0p', 0.05:'dpatterns/noise/5p', 
               0.15:'dpatterns/noise/15p', 0.25:'dpatterns/noise/25p', 
               0.4:'dpatterns/noise/40p', 0.5:'dpatterns/noise/50p', 
               0.75:'dpatterns/noise/75p', 1 :'dpatterns/noise/100p'}

    folders = {0 :'patterns/noise/0p', 0.05:'patterns/noise/5p', 
               0.15:'patterns/noise/15p', 0.25:'patterns/noise/25p', 
               0.4:'patterns/noise/40p', 0.5:'patterns/noise/50p', 
               0.75:'patterns/noise/75p', 1 :'patterns/noise/100p'}

    resize_wh = (20, 20) 
    # d folders image process
    dresize_wh = (1000, 1000)
    imp.resize_patterns(training, resized, resize_wh)
    imp.binarize_patterns(resized, binarized)

    for percent in folders:
        imp.create_noisy_patterns(binarized, folders[percent],  percent)

    for key in folders:
        imp.resize_patterns(folders[key], dfolders[key], dresize_wh) 

if __name__ == '__main__':
    main()

