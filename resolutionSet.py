import sys
import argparse
import cv2
import os
from os import listdir
from os.path import isfile,join
"""
example usage:
    converting a directory of photos:
        python3 resolutionSet.py --pathIn photos --pathOut photosResAdjusted
    converting a video to a directory of photos (must be mp4):
        python3 resolutionSet.py --pathIn ./video/video.mp4 --pathOut photosResAdjusted --fps 5
        fps 10 samples every 5th frame 
"""

def extractImages(pathIn, pathOut,fps=10):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    fpscountIt=fps 
    while success:
        #1fs
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,count)    # added this line
        count = count + fpscountIt
        success,image = vidcap.read()
        fNameNew=pathOut+os.path.sep+"frame{0}".format(count)+".bmp"
        if image is None:
            continue
        try:
            cv2.imwrite(str(os.path.join(os.getcwd(),fNameNew)), cv2.resize(image,(424,240)))     # save frame as JPEG file
        except:
            continue

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video or photos original")
    a.add_argument("--pathOut", help="path to images")
    a.add_argument("--fps",default=False,type=float,help="set fps for video sample")
    args = a.parse_args()
    if os.path.splitext(args.pathIn)[-1].lower()==".mp4":
        extractImages(args.pathIn, args.pathOut,args.fps)
    else:
        filesList = [f for f in listdir(args.pathIn) if isfile(join(args.pathIn,f))]
        for filesListIterator in filesList:
            im = cv2.imread(os.path.join(os.getcwd(),args.pathIn,filesListIterator))
            cv2.imwrite(os.path.join(os.getcwd(),args.pathOut,filesListIterator), cv2.resize(im,(424,240)))
         
