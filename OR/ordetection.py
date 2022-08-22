# import directory with neutral network

from OR.Yolo.yolov3.utils import *
from OR.Mobile_detection.mobile_detection import *

class ordetection():
    # When initiating the orbject class, it is required to specify the type of neural network you want to use

    def __init__(self, type):
        if type == 'yolo':
            # run YOLO neutral network
            self.neutralNetwork = Load_Yolo_model()
        else:
            # run SSD MobileNet neutral network
            if type == 'MobileNet':
                self.neutralNetwork = start_network()

    def detection(self, img, type, img_id):
        # return list of box object
        if type == 'yolo':
            return detect_frame(self.neutralNetwork, img, 416, img_id)
        if type == 'Mobile':
            return detection(img)
