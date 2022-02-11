import torch 
import pandas
import os
from os import listdir
from os.path import isfile,join
import cv2 
import numpy as np
import argparse
import pandas as pd
'''
example usage:
    python3 detectDir.py --photos "photos" --threshold 0.8 --outDir results --model weights/weights.pt --show True
packages:
    pip3 install opencv-python
    pip3 install torch
    pip3 install tqdm
    pip3 install torchvision
    pip3 install seaborn
    ultralytics should then install missing modules
run directory setup:
    detectDir.py
    photos - photo1.jpg,photo2.jpg
    results
    weights - weights.pt
tip:
    rm results*.csv to remove old result files
'''
def getInputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p','--photos',
                        default=False, 
                        type=str, 
                        help='photos directory to be inputted'
                        )
    parser.add_argument('-t','--threshold',
                        default=False, 
                        type=float, 
                        help='baudrate that esp8266 is transfering data at'
                        )
    parser.add_argument('-o','--outDir',
                            default=False,
                            type=str,
                            help='output directory of images')
    parser.add_argument('-m','--model',
                            default=False,
                            type=str,
                            help='location of weights.pt')
    parser.add_argument('-s','--show',
                            default = False,
                            type=bool,
                            help='output images')

    args = parser.parse_args()
    #get user inputs
    photos = args.photos
    threshold = args.threshold
    outputDir=args.outDir
    modelFile = args.model
    show = args.show
    return photos, threshold,outputDir,modelFile, show

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def saveVisual(numpyImage,resultsDf,resultDir,ID):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches 
        fix,ax = plt.subplots()
        ax.imshow(numpyImage)
        for index,row in resultsDf.iterrows():
            xLen=row['xmax']-row['xmin']
            yLen=row['ymax']-row['ymin']
            rect=patches.Rectangle((row['xmin'],row['ymin']),xLen,yLen,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.savefig(os.path.join(resultDir,"objDetection{1}.jpg".format(resultDir,ID)))

def main():
    directory,scoreThreshold,resultsDirectory,BESTMODELPATH,show=getInputs() 
    MODELPATH="ultralytics/yolov5"
    model = torch.hub.load(MODELPATH,'custom',path=BESTMODELPATH)
    filesList = [f for f in listdir(directory) if isfile(join(directory,f))]
    resultsDict={"fname":[]}
    #print("number of images being tested is ", len(filesList))
    for filesListIterator in filesList:
        tmpResults={filesListIterator : []}
        im = cv2.imread(os.path.join(os.getcwd(),directory,filesListIterator))
        objsDf=model(im).pandas().xyxy[0]
        #print(objsDf)
        objsDf=objsDf[(objsDf[["confidence"]]>=scoreThreshold).all(1)]
        if len(objsDf) == 0:
            #print(filesListIterator,"no bottle detected...")
            continue
        #print(filesListIterator, "is above threshold with result: ")
        if show == True and resultsDirectory is not None:
            saveVisual(im,objsDf,resultsDirectory,filesListIterator)
        for index,row in objsDf.iterrows():
            tmpResults[filesListIterator].append(row["confidence"])
            print(filesListIterator,",",row["confidence"],",")
            tmpResults[filesListIterator].append(row["xmin"])
            tmpResults[filesListIterator].append(row["xmax"])
            tmpResults[filesListIterator].append(row["ymin"])
            tmpResults[filesListIterator].append(row["ymax"])
        resultsDict["fname"].append(tmpResults)
    if len(resultsDict["fname"])>0:
        pd.DataFrame.from_dict(resultsDict).to_csv(uniquify(str(os.path.join(os.getcwd(),"results.csv"))))
    else:
        print("no bottles detected from any input images...")
if __name__=="__main__":
    main()
