from django.apps import AppConfig


class LivefeedConfig(AppConfig):
    ran_already = False
    def ready(self):  
        if LivefeedConfig.ran_already == False:     
            LivefeedConfig.ran_already = True
            from .VideoCamera import VideoCamera        
            print("You only YOLO once")
            self.camera = VideoCamera() 
            
    name = 'livefeed'
    