import numpy as np  # Scientific computing library
import sys, json, os, time, datetime, cv2
import box

# more information and example data
# https://automaticaddison.com/how-to-detect-objects-in-video-using-mobilenet-ssd-in-opencv/
# https://drive.google.com/drive/folders/13nAcSOx_S8QkbDw_oEimv0yRyfJazUNG
# https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb

classes, colors = [], []
neural_network = None
RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255


# List of categories and classes

# Create the bounding boxes
# bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))
def detection(frame):
    global neural_network, colors, classes
    (h, w) = frame.shape[:2]
    list_box = list()
    file_size = (w, h)
    # Create a blob. A blob is a group of connected pixels in a binary
    # frame that share some common property (e.g. grayscale value)
    # Preprocess the frame to prepare it for deep learning classification
    frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS),
                                       IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
    # Set the input for the neural network
    neural_network.setInput(frame_blob)
    # Predict the objects in the image
    neural_network_output = neural_network.forward()
    # Put the bounding boxes around the detected objects
    list_obj = []
    for i in np.arange(0, neural_network_output.shape[2]):
        confidence = neural_network_output[0, 0, i, 2]
        # Confidence must be at least 30%
        if confidence > 0.35:
            idx = int(neural_network_output[0, 0, i, 1])
            bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = bounding_box.astype("int")
            bbox_colors = colors[idx]
            label = classes[idx]
            b = box.BoundBox(startX, startY, endX, endY)
            b.classes = idx
            b.score = confidence
            b.color = bbox_colors
            b.v_label = label
            list_box.append(b)
    return list_box

def start_network():
    global neural_network, classes, colors

    # Load the pre-trained neural network
    neural_network = cv2.dnn.readNetFromCaffe('model/MobileNetSSD_deploy.prototxt.txt',
                                              'model/MobileNetSSD_deploy.caffemodel')
    # read label from file in "model" directory
    with open("model/label.txt", "r") as label_file:
        for l in label_file.readlines():
            classes.append(l)

    # if constant colors are required they can be loaded from file
    with open("model/color.txt", "r") as color_file:
        for c in color_file.readlines():
            temp_color = c.split(",")
            colors.append([temp_color[0], temp_color[1], temp_color[2]])
