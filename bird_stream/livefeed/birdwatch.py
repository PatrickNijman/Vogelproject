# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:00:07 2020

@author: Pat
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
from keras.models import load_model

vert =256
compf = 0.5

class Main:
    
    def __init__(self, Threaded = False):
        self.counter = 0
        self.waiting_for_threads = True
        self.sending = False
        self.transfer = False
        self.threads = []
        self.closeups = []
        self.closeuptime = []
        self.fullimgs = []
        self.fullimgstime = []
        conn = sqlite3.connect('Captures.db')
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        self._table_check()
        self.BirdCam = BirdCamera()
        self.watch_birds()     
        self.print_db()
        
    def watch_birds(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame' , 600 , 400)  
        while True:
            ret, frame = self.BirdCam.cap.read()
            self.BirdCam.processimages(frame)
            image_copy = frame.copy()
            self._prepdata(image_copy)
            for location in self.BirdCam.objectscurrentframe:
                cv2.rectangle(frame , ( int(location[0] - vert / 4), int(location[1] - vert / 4)), (int(location[0] + vert / 4),  int(location[1] + vert / 4)), (255, 255, 255))
            cv2.imshow('frame', frame)
            self._imgmanager()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._imgmanager(closing = True)
                break
        cv2.destroyAllWindows()
    
    def _prepdata(self, frame):
        if len(self.BirdCam.objectscurrentframe) > 0:
            
            date_and_time = datetime.datetime.now()
            #cv2.putText(frame, str(date_and_time ), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255, 255), 1, cv2.LINE_AA)
            self.fullimgs.append(frame)
            self.fullimgstime.append(date_and_time)
        else:
            return None
        
        for location in self.BirdCam.objectscurrentframe:
            im = self.BirdCam._cropper(frame, location[0], location[1])
            self.closeups.append(im)
            self.closeuptime.append(date_and_time)
        return None
            
    def _imgmanager(self, closing = False):

#         if self.counter > 5:
#             self.transfer = True
#             self.counter = 0
        if (len(self.closeups) > 10 or closing is True) and self.transfer is False:
            self.counter += 1
            t = threading.Thread(target = self._send_data, args = (self.closeups, self.closeuptime, self.fullimgs, self.fullimgstime,))
            t.start()
            self.threads.append(t)
            self.closeups = []
            self.closeuptime = []
            self.fullimgs = []
            self.fullimgstime = []
        elif self.transfer is True and self.sending is False:
            self.sending = True
            print('Waiting for threads to finish')
            transfer_thread = threading.Thread(target = self._transfer_data)
            transfer_thread.start()
        return 

    def _table_check(self):
        conn = sqlite3.connect('Captures.db')
        
        sql_create_fullimages_table = """ CREATE TABLE IF NOT EXISTS fullimages (
                                        date STRING NOT NULL,
                                        format STRING NOT NULL,
                                        image BLOB NOT NULL
                                    ); """
    
        sql_create_closeups_table = """ CREATE TABLE IF NOT EXISTS closeups (
                                        date STRING NOT NULL,
                                        format STRING NOT NULL,
                                        image BLOB NOT NULL
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_fullimages_table)
        c.execute(sql_create_closeups_table)
        conn.close()
        return None
 
    
 
    def _transfer_data(self):
        if self.threads == 0:
            print('transfering data')
            # Send Captures.db to computer here
            os.remove('Captures.db')
            conn = sqlite3.connect('Captures.db')
            conn.execute("PRAGMA journal_mode=WAL")
            conn.close()
            self._table_check()
            self.sending = False
            print('data transfered')
            self.transfer = False
            self.waiting_for_threads = True
        elif self.waiting_for_threads == True:
            t = threading.Thread(target =self._joinandkill)
            t.start()
            time.sleep(0.5)
            self._transfer_data()
        else:
            time.sleep(0.5)
            self._transfer_data()
        return 
    
    def _joinandkill(self):
        self.waiting_for_threads = False
        for n in range(len(self.threads)-1):
            print('{} threads left'.format(n+1))
            t = self.threads.pop(0)
            t.join()
        return
    
    def _send_data(self, closeups, closeuptime, fullimgs, fullimgstime):
        conn = sqlite3.connect('Captures.db')
        for date, closeup in zip(closeuptime, closeups):
            query = """ INSERT INTO closeups VALUES(?, ?, ?)"""
            cur = conn.cursor()
            cu_asbytes = pickle.dumps(closeup)
            cur.execute(query, (str(date), str(closeup.shape), cu_asbytes))
  
        for date, img in zip(fullimgstime, fullimgs):
            query = """ INSERT INTO fullimages VALUES(?, ?, ?)"""
            cur = conn.cursor()
            img_asbytes = pickle.dumps(img)
            cur.execute(query, (str(date), str(img.shape), img_asbytes))
        conn.commit()
#         cur.execute("select date from fullimages")
#         results = cur.fetchall()
#         row_count = len(results)
        conn.close()
        return #srow_count
    
    def print_db(self):
        conn = sqlite3.connect('Captures.db')   
        cur = conn.cursor()
        cur.execute("SELECT image FROM fullimages")
        rows = cur.fetchall()
        for row in rows:
            img = pickle.loads(row[0])
            while True:
                cv2.imshow('frame', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
            cv2.destroyAllWindows()
        conn.close()
        return None
    

class ImgCommunicator:
    
    def __init__(self, ip, mode, port=5555, clients = 1):
        self.ip_connect = ip
        self.port_connect = port
        self.mode = mode
        self.connected = False
        self.sendthread = None
        self.recvthread = None
        self.sock = socket.socket()#socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket = None
        self.img = None
        self.stop = False
        self.new_img_flag = False
        if mode == 'server':
            self.sock.bind((ip, port))
            self.sock.listen(0)
            self.clientsocket, _ = self.sock.accept()
            self._startrecv()     
        elif mode == 'client':
            self.sock.connect((ip, port))
    
    def get_image(self):
        if self.new_img_flag is False:
            time.sleep(0.1)
            return self.get_image()
        else:
            self.new_img_flag = False
            return self.img
    
    def send(self, data):
        if self.sendthread is not None:
            self.sendthread.join()
        self.sendthread = threading.Thread(target= self.socket.sendall, args = (data,)) 
        self.sendthread.start()    
        return True
    
    def receive(self):
        while True:
            if self.new_img_flag is False:
                msg = self.clientsocket.recv(4000000000)
                self.img = msg   
                self.new_img_flag = True
            if self.stop:
                break
        return 
    
    def _startrecv(self):
        self.recvthread = threading.Thread(target = self.receive)
        self.recvthread.start()
    
    def close(self):
        self.stop = True
        if self.mode == 'client':
            self.sendthread.join()
        elif self.mode == 'server':
            self.recvthread.join()
        self.sock.close()
        
        
class BirdCamera:
    global vert, compf
    
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=50, backgroundRatio=0.01, nmixtures=3, noiseSigma=0)
        self.selem_open = skimage.morphology.disk(5, dtype=np.uint8)
        self.fgmask = None
        self.detector = self._blobdetinit()
        self.imdil = None
        self.objectscurrentframe = []
        self.triggercounter = 0
        self.lasttriggercount = 0
        self.writer = None
        # self.model = load_model('my_model_fc__birds.hdf5')
        
    def look_for_birds(self, image):
        # im = cv2.resize(image, (384, 216))/255
        # im = im.reshape((1,) + im.shape)
        # pr = self.model.predict(im)
        return #False if pr[0][0] == 0 else True
    
    def processimages(self, frame):   
        small_frame = self._reduceframe(frame)
        self._foregrounddetector(small_frame)
        openimg = self._imagedilation(self.fgmask)
        self._objectdetector(openimg)
        return None
    
    def _cropper(self,fullframe, x, y): #location is x,y data(integer)
        # Crops part of the image, the cropping size is suitable for reconstruction of multiple depths.
        location = (x, y)
        crop = fullframe[int(location[1] - vert / 2): int(location[1] + vert / 2), int(location[0] - vert / 2): int(location[0] + vert / 2)]
        return crop
    
    def _reduceframe(self, fullframe):
        # Reduces the size of the input frame in order to reduce computational time during foreground detection and blob detection
        smallframe = cv2.resize(fullframe, (0, 0), fx=compf, fy=compf)
        smallframe= cv2.cvtColor(smallframe, cv2.COLOR_BGR2GRAY)
        return smallframe

    def _foregrounddetector(self, frame):
        # Updates the foreground mask to the current frame
        self.fgmask = self.fgbg.apply(frame, 0.05)

    def _imagedilation(self,frame):
        # Dilates the image in order to magnify small objects in the foreground mask
        self.imdil = skimage.morphology.opening(frame, selem=self.selem_open, out=None)
        return self.imdil

    def _objectdetector(self, frame):
        # Detects white blobs in the image, and appends the location and frame number to a global list of the class.
        kp = self.detector.detect(frame)
        self.objectscurrentframe = []
        fsize = frame.shape
        for point in kp:
            location = (int(point.pt[0]/compf), int(point.pt[1]/compf))
            if vert/2 < location[1] < (fsize[0]/compf - vert/2)  and  vert/2 < location[0] < (fsize[1]/compf - vert/2):
                self.objectscurrentframe.append(location)
    
    def trigger(self):
        self.lasttriggercount = self.triggercounter
        if self.triggercounter > 0:
            self.triggercounter += -1
        if len(self.objectscurrentframe) > 0 and self.triggercounter <= 5:
            self.triggercounter = 10
        return None
        
    def videowriter(self, frame, date = None):
        if self.lasttriggercount == 0 and self.triggercounter > 0:
            path = 'C:/Users/Pat/.spyder-py3/ImageData/videos/'
            output_params = {"-vcodec":"libx264", "-input_framerate":3} #"-crf": 0, "-preset": "fast", "-tune": "zerolatency",
            self.writer = WriteGear(output_filename = os.path.join(path , '{}.mp4'.format(date)), **output_params)
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
    
class DBmanager:
    
    def __init__(self, Threaded = False):
        self.counter = 0
        self.waiting_for_threads = True
        self.sending = False
        self.transfer = False
        self.threads = []
        self.closeups = []
        self.closeuptime = []
        self.fullimgs = []
        self.fullimgstime = []
        conn = sqlite3.connect('Captures.db')
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        self._table_check()

    
    def _prepdata(self, frame, birdcam):
        if len(birdcam.objectscurrentframe) > 0: 
            date_and_time = datetime.datetime.now()
            # cv2.putText(frame, str(date_and_time ), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255, 255), 1, cv2.LINE_AA)
            self.fullimgs.append(frame)
            self.fullimgstime.append(date_and_time)
            for location in birdcam.objectscurrentframe:
                im = birdcam._cropper(frame, location[0], location[1])
                self.closeups.append(im)
                self.closeuptime.append(date_and_time)
            return None
        return None
        
            
    def _imgmanager(self, closing = False):
        if len(self.fullimgs) > 10 or closing is True:
            t = threading.Thread(target = self._send_data, args = (self.closeups, self.closeuptime, self.fullimgs, self.fullimgstime,))
            t.start()
            self.threads.append(t)
            self.closeups = []
            self.closeuptime = []
            self.fullimgs = []
            self.fullimgstime = []
        return None

    def _table_check(self):
        conn = sqlite3.connect('Captures.db')
        
        sql_create_fullimages_table = """ CREATE TABLE IF NOT EXISTS fullimages (
                                        date STRING NOT NULL,
                                        format STRING NOT NULL,
                                        image BLOB NOT NULL
                                    ); """
    
        sql_create_closeups_table = """ CREATE TABLE IF NOT EXISTS closeups (
                                        date STRING NOT NULL,
                                        format STRING NOT NULL,
                                        image BLOB NOT NULL
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_fullimages_table)
        c.execute(sql_create_closeups_table)
        conn.close()
        return None
    
    
    def _send_data(self, closeups, closeuptime, fullimgs, fullimgstime):
        print('Sending data to server')
        conn = sqlite3.connect('Captures.db')
        for date, closeup in zip(closeuptime, closeups):
            query = """ INSERT INTO closeups VALUES(?, ?, ?)"""
            cur = conn.cursor()
            cu_asbytes = pickle.dumps(closeup)
            cur.execute(query, (str(date), str(closeup.shape), cu_asbytes))
  
        for date, img in zip(fullimgstime, fullimgs):
            query = """ INSERT INTO fullimages VALUES(?, ?, ?)"""
            cur = conn.cursor()
            img_asbytes = pickle.dumps(img)
            cur.execute(query, (str(date), str(img.shape), img_asbytes))
        conn.commit()
#         cur.execute("select date from fullimages")
#         results = cur.fetchall()
#         row_count = len(results)
        conn.close()
        print('Finished sending data')
        return #srow_count
    
    def print_db(self):
        conn = sqlite3.connect('Captures.db')   
        cur = conn.cursor()
        cur.execute("SELECT image FROM fullimages")
        while True:
            row = cur.fetchone()
            if row is None:
                break
            img = pickle.loads(row[0])
            while True:
                cv2.imshow('frame', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
            cv2.destroyAllWindows()
        conn.close()
        return None
    
    def pop_image(self, table = 'closeups'):
        conn = sqlite3.connect('Captures.db')   
        cur = conn.cursor()
        cur.execute("SELECT image, date FROM {} LIMIT 1".format(table))
        try:
            row, date = cur.fetchone()
        except:
            return None, None
        img = pickle.loads(row)
        #cv2.putText(img, str(date), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255, 255), 1, cv2.LINE_AA)
        cur.execute("DELETE FROM {} WHERE date = ?".format(table), (date,))
        conn.commit()
        conn.close()
        return img, date
    
if __name__ == '__main__':
    Main()
