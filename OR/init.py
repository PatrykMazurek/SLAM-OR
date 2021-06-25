# import directory with neutral network

import Yolo.yolov3.utils as yolo
import Mobile_detection.mobile_detection as mobileNet


# jeżeli potrzebna jest incjalizacja sieci neuronowej wykonaj incjalizację i zwróć obiekt sici neuronowej

neutralNetwork = ""

def start_network(type):
    global neutralNetwork
    # run YOLO neutral network
    if type == 'Yolo':
        neutralNetwork = yolo.Load_Yolo_model()
    # run SSD MobileNet neutral network
    if type == 'MobileNet':
        neutralNetwork = mobileNet.main()

def detection(img, type, img_id):
    # return list of box object
    if type == 'Yolo':
        return yolo.detect_frame(neutralNetwork, img, 416, img_id)
    if type == 'Mobile':
        return mobileNet.detection(img)
