from datetime import datetime
from vidgear.gears import NetGear
import numpy as np
import imagezmq
import imutils
import cv2
import skimage.morphology 
import skimage.feature
from birdwatch import BirdDetector
from threading import Thread
import zstandard as zstd
import blosc
import socket
import base64 as b64
vert =256
compf = 1

BC = BirdDetector(classify=True)

c = 0.9
options = {'flag' : 0, 'copy' : False, 'track' : False, 'compression_param':cv2.IMREAD_COLOR}

client = NetGear(address = '192.168.2.71', port = '5555', protocol = 'tcp',  pattern = 0, receive_mode = True, logging = True, **options)


while True:
  
    t1 = datetime.now()
    frame = client.recv()
    if frame is None:
        break
    t2 = datetime.now()
    diff = t2 - t1

    t3 = datetime.now()
    diff = t3 - t2
    newfr = BC.processimages(frame)
    BC.videowriter(frame, date= str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    cv2.imshow("BirdCam", np.concatenate((BC.fgmask,BC.imdil, BC._reduceframe(frame)),axis=0))#frame) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print('Exiting')
        print('Finished streaming')
        client.close()
        break
# do a bit of cleanup
cv2.destroyAllWindows()
