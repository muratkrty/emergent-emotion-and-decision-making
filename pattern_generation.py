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

    folders = {0.05:'patterns/noise/5p', 0.15:'patterns/noise/15p',
               0.25:'patterns/noise/25p', 0.4:'patterns/noise/40p',
               0.5:'patterns/noise/50p', 0.75:'patterns/noise/75p', 
               1 :'patterns/noise/100p'}

    resize_wh = (20, 20)
    imp.resize_patterns(training, resized, resize_wh)
    imp.binarize_patterns(resized, binarized)

    for percent in folders:
        imp.create_noisy_patterns(binarized, folders[ percent],  percent)

if __name__ == '__main__':
    main()

