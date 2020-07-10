# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:00:07 2020

@author: Pat
changes: removed all database related functions
"""
import os
import numpy as np
import pickle
import cv2
import datetime
import threading
import skimage.morphology
import skimage.feature
import sys
from vidgear.gears import WriteGear
import time
import sqlite3
from queue import Queue
import socket
import names
from keras.models import load_model

vert =300
compf = 0.5

class BirdDetector:
    global vert, compf
    
    def __init__(self, classify = False, T = 0.95):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=50, backgroundRatio=0.05, nmixtures=3, noiseSigma=0)
        self.selem_open = skimage.morphology.disk(3, dtype=np.uint8)
        self.fgmask = None
        self.detector = self._blobdetinit()
        self.imdil = None
        self.classes = ['House sparrow', 'Great tit', 'Eurasian blue tit', 'Eurasian magpie', 'Eurasian jay']
        self.objectscurrentframe = []
        self.objectslastframe = self.objectscurrentframe
        self.triggercounter = 0
        self.lasttriggercount = 0
        self.birdcount = 0
        self.writer = None
        if classify is True:
            self.model = load_model('my_model_fc__birds.hdf5')
            self.threshold = T
        else:
            self.model = None
            
    def processimages(self, frame):   
        small_frame = self._reduceframe(frame)
        self._foregrounddetector(small_frame)
        openimg = self._imagedilation(self.fgmask)
        obj = self._objectdetector(openimg)
        self._look_for_birds(frame, obj)
        self._crossreference()
        self.trigger()
        for location in self.objectscurrentframe:
            cv2.rectangle(frame , ( int(location[0] - vert / 4), int(location[1] - vert / 4)), (int(location[0] + vert / 4),  int(location[1] + vert / 4)), (255, 255, 255))
            cv2.putText(frame, location[2] +': {}'.format(location[3]), (int(location[0] - vert / 4), int(location[1] + vert / 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)
        return frame
        
    def _look_for_birds(self, frame, locations):
        self.objectslastframe = self.objectscurrentframe
        self.objectscurrentframe = []
        if len(locations) == 1:
            crop = self._cropper(frame, locations[0][0], locations[0][1])
            im = cv2.resize(crop, (96, 96))/255
            im = im.reshape((1,) + im.shape)
            if self.model is not None:
                pr = self.model.predict(im)
                name = self.classes[np.argmax(pr[0])] if max(pr[0]) > self.threshold else None
                if name is not None:
                    print(name)
                    self.objectscurrentframe.append([locations[0][0], locations[0][1], name])
            else:
                name = 'placeholder bird'
                self.objectscurrentframe.append([locations[0][0], locations[0][1], name])
        else:
            for location in locations:
                crop = self._cropper(frame, location[0], location [1])
                im = cv2.resize(crop, (96, 96))/255
                im = im.reshape((1,) + im.shape)
                if self.model is not None:
                    pr = self.model.predict(im)
                    name = self.classes[np.argmax(pr[0])] if max(pr[0]) > self.threshold else None
                    if name is not None:    
                        print(name)
                        self.objectscurrentframe.append([location[0], location[1], name])
                else:
                    name = 'placeholder bird'
                    self.objectscurrentframe.append([location[0], location[1], name])
        return 
    
    def _cropper(self,fullframe, x, y): #location is x,y data(integer)
        # Crops part of the image, the cropping size is suitable for reconstruction of multiple depths.
        crop = fullframe[int(y - vert / 2): int(y + vert / 2), int(x - vert / 2): int(x + vert / 2)]
        return crop
    
    def _reduceframe(self, fullframe):
        # Reduces the size of the input frame in order to reduce computational time during foreground detection and blob detection
        smallframe = cv2.resize(fullframe, (0, 0), fx=compf, fy=compf)
        smallframe= cv2.cvtColor(smallframe, cv2.COLOR_BGR2GRAY)
        return smallframe

    def _foregrounddetector(self, frame):
        # Updates the foreground mask to the current frame
        self.fgmask = self.fgbg.apply(frame, 0.01)

    def _imagedilation(self,frame):
        # Dilates the image in order to magnify small objects in the foreground mask
        self.imdil = skimage.morphology.opening(frame, selem=self.selem_open, out=None)
        return self.imdil

    def _objectdetector(self, frame):
        # Detects white blobs in the image, and appends the location and frame number to a global list of the class.
        kp = self.detector.detect(frame)
        objectsfound = []
        fsize = frame.shape
        for point in kp:
            location = (int(point.pt[0]/compf), int(point.pt[1]/compf))
            if vert/2 < location[1] < (fsize[0]/compf - vert/2)  and  vert/2 < location[0] < (fsize[1]/compf - vert/2):
                objectsfound.append(location)
        return objectsfound     
    
    def _crossreference(self):
        T = 120
        for n in range(len(self.objectscurrentframe)):
            self.objectscurrentframe[n].append(None)
        if len(self.objectslastframe) == 0 or len(self.objectscurrentframe) == 0:
            pass
        else:
            mat = np.empty((len(self.objectslastframe),len(self.objectscurrentframe)))
            for n in range(len(self.objectslastframe)):
                for m in range(len(self.objectscurrentframe)):
                        mat[n,m] = np.sqrt((self.objectscurrentframe[m][0]-self.objectslastframe[n][0])**2  +  (self.objectscurrentframe[m][1]-self.objectslastframe[n][1])**2)
            n = 0
            m = min([len(self.objectslastframe), len(self.objectscurrentframe)])
            while True:
                ind = np.unravel_index(np.argmin(mat, axis=None), mat.shape)
                if mat[ind] < T and self.objectscurrentframe[ind[1]][2] is self.objectslastframe[ind[0]][2] and self.objectscurrentframe[ind[1]][2] is not None:
                    self.objectscurrentframe[ind[1]][3] = self.objectslastframe[ind[0]][3]
                elif mat[ind] < T and self.objectscurrentframe[ind[1]][2] is not self.objectslastframe[ind[0]][2]:
                     n+=-1
                     mat[ind] = np.inf
                mat[:,ind[1]] = np.inf
                mat[ind[0],:] = np.inf
                n+=1
                if n == m:
                    break
        for m in range(len(self.objectscurrentframe)):
            if self.objectscurrentframe[m][3] is None and self.objectscurrentframe[m][2] is not None:
                self.objectscurrentframe[m][3] = names.get_first_name()
                self.birdcount += 1
                print('new {} found: {}'.format(self.objectscurrentframe[m][2],self.objectscurrentframe[m][3]))             
        return None
        
    def trigger(self):
        self.lasttriggercount = self.triggercounter
        if self.triggercounter > 0:
            self.triggercounter += -1
        if len(self.objectscurrentframe) > 0 and self.triggercounter <= 10:
            self.triggercounter = 20
        return None
        
    def videowriter(self, frame, date = None):
        if self.lasttriggercount == 0 and self.triggercounter > 0:
            path = 'C:/Users/Pat/.spyder-py3/ImageData/videos/'
            output_params = {"-fourcc":"MJPG", "-fps": 8} #"-crf": 0, "-preset": "fast", "-tune": "zerolatency",
            self.writer = WriteGear(output_filename = os.path.join(path , '{}.mp4'.format(date)), compression_mode=False, **output_params)
            self.writer.write(frame)
            #start video recording
            return None
        elif self.lasttriggercount > 0 and self.triggercounter > 0:
            self.writer.write(frame)
            #keep recording
            return None
        elif self.lasttriggercount > 0 and self.triggercounter ==0:
            self.writer.write(frame)
            self.writer.close()
            self.writer = None
            #close recording
            return None
        else:
            return None
    
    def _blobdetinit(self, minT=10, maxT=50,fbA = False, minA = 3, maxA = 2000, fbC = False, minC = 0, fbI = False, minI = 0.01, fbCol = False, fbCon = False):
        # Used to initialize and tune the blob detector.
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = minT
        params.maxThreshold = maxT
        params.filterByArea = fbA
        params.minArea = minA
        params.maxArea = maxA
        params.filterByCircularity = fbC
        params.minCircularity = minC
        params.filterByInertia = fbI
        params.minInertiaRatio = minI
        params.filterByColor = fbCol
        params.filterByConvexity = fbCon
        params.minConvexity = 0.01
        params.blobColor = 0
        params.minDistBetweenBlobs = 50
        blobdetector = cv2.SimpleBlobDetector_create(params)
        return blobdetector

    
if __name__ == '__main__':
    import server