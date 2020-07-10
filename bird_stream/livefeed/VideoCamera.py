# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:45:51 2020

@author: Pat
"""
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip
from vidgear.gears import NetGear
from .birdwatch import BirdDetector, DBmanager, ImgCommunicator
import cv2
import numpy as np
import threading
import time

class VideoCamera(object):
    def __init__(self):
        self.BC = BirdCamera()
        options = {'flag' : 0, 'copy' : False, 'track' : False, 'compression_param':cv2.IMREAD_COLOR}
        self.client = NetGear(address = '192.168.2.71', port = '5555', protocol = 'tcp',  pattern = 0, receive_mode = True, logging = True, **options)
        self.grabbed = True
        self.frame = self.client.recv()
        self.lock = threading.Lock()
        threading.Thread(target=self.update, args=()).start()
        
        
    def __del__(self):
        self.client.close()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                image = self.frame
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
            else:
                return None

    def update(self):        
        while True:
            with self.lock:
                frame = self.client.recv()
                self.BC.processimages(frame)
                # for location in self.BC.objectscurrentframe:
                #       cv2.rectangle(frame , ( int(location[0] - vert / 4), int(location[1] - vert / 4)), (int(location[0] + vert / 4),  int(location[1] + vert / 4)), (255, 255, 255))
                concat_frame = np.concatenate((self.BC.fgmask,self.BC.imdil, self.BC._reduceframe(frame)),axis=0)
                self.frame = concat_frame
                if self.frame is not None:
                    self.grabbed = True
                else: 
                    self.grabbed = False



