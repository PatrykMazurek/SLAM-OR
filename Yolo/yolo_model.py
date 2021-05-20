from yolov3 import utils
from config import *

# jeżeli programy mają działać oddzielnie to wywołaj odpowiednie metody
# Działanie oddzielne YOLO zapisuje rekordy do pliku json a następnie SLAM odczytuje dane z pliku
# i odpowiednio porównuje ze swoimi wynikami. format pliku json:
# {frame : nr [{obj : name ; x_min : value; y_min : value; x_max : value; y_max : value]},

# działanie jednoczesne
# zwracany jest obiek YOLO do Programu SLAM i rozpoznawanie obiektów jednoczeńnie.

def YoloWork(save_to_file):
    if save_to_file:
        print("zapisuje dane do pliku json")
        # zapisuje rekordy do pliku json
    else:
        return utils.detect_image()
