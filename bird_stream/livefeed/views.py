from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip
import cv2
import threading
import time
from django.apps import apps
# from ..bird_stream.VideoStream import VideoStream
# Create your views here.
 
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         self.lock = threading.Lock()
#         threading.Thread(target=self.update, args=()).start()
        
        
#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         my_app_config = apps.get_app_config('livefeed')
#         req = my_app_config.test
        
#         with self.lock:
#             if self.frame is not None:
#                 image = self.frame
#                 ret, jpeg = cv2.imencode('.jpg', image)
#                 return jpeg.tobytes()
#             else:
#                 return None

#     def update(self):        
#         while True:
#             with self.lock:
#                 (self.grabbed, self.frame) = self.video.read()

# class VideoStream:
#     def __init__(self, src=0):
#         # initialize the video camera stream and read the first frame
#         # from the stream
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         # initialize the variable used to indicate if the thread should
#         # be stopped
#         self.count = 0
#         self.stopped = False
        
#     def start(self):
#         # start the thread to read frames from the video stream
#         Thread(target=self.update, args=()).start()
#         return self
    
#     def update(self):
#         # keep looping infinitely until the thread is stopped
#         while True:
#             # if the thread indicator variable is set, stop the thread
#             if self.stopped:
#                 return
#             # otherwise, read the next frame from the stream
#             (self.grabbed, self.frame) = self.stream.read()
            
#     def read(self):
#         # return the frame most recently read
#         return self.frame
    
#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        



@gzip.gzip_page
def livefe(request):
    my_app_config = apps.get_app_config('livefeed')
    camera = my_app_config.camera    
    try:
        return StreamingHttpResponse(gen(camera), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def home(request):
    return render(request, 'livefeed/home.html', {'title': 'live bird view'})

    
