# Author: Murat Kirtay, The Biorobotics Inst./SSSA
# Description: Yarp provides camera images in .ppm format and located
#              in ~ directory

import os

def move_ppms(frames):
    '''Copy and rename ppm files to the project directory '''
    for i in range(len(frames)):
        cmd = 'cp ~/' + frames[i] + '  ./' + str(i) + '.ppm'
        os.system(cmd)

def main():

    # Copy .ppm files, to project directory 
    # Frame names do not change, use a list
    frames = ['frame000.ppm', 'frame001.ppm', 'frame002.ppm', \
               'frame003.ppm', 'frame004.ppm']

    move_ppms(frames)

if __name__ == '__main__':
    main()
