# based on https://github.com/experiencor/keras-yolo3

import numpy as np

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        self.color = []
        self.v_label = ""
        self.pose = []

    def __str__(self):
        return "{} {} {} {} {} {}".format(self.v_label, self.pose, self.xmin, self.xmax, self.ymin, self.ymax)

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

    def get_csv_point_with_camera_pose(self, frame_id, camera):
        t = "{};{};{}\n".format(frame_id, self.pose, camera)
        return t

    # frame_nr; xmin; ymin; xmax; ymax; score; class_ind; R; G; B; label; pose
    def get_csv_format(self, frame_id):
        t = "{};{};{};{};{};{};{};{};{};{};{};{}\n".format(frame_id, self.xmin, self.ymin, self.xmax, self.ymax,
                                            self.score, self.classes,
                                            self.color[0], self.color[1], self.color[2],
                                            self.v_label, self.pose)
        return t

    def get_json_format(self):
        t = {"xmin" : int(self.xmin), "ymin" : int(self.ymin), "xmax" : int(self.xmax), "ymax" : int(self.ymax),
               "score" : self.score, "class_ind" : int(self.classes), "color" : self.color,
             "label": self.v_label, "pose" : self.pose}
        return t

