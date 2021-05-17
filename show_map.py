import numpy as np
import os, cv2, json, math, time
import OpenGL.GL as gl
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

# this file present date from json file and csv file o nterh map 3D
# and calculate the position of opints using DBSCAN and present in map
import viewer3D
from viewer3D import Viewer3D

viewer = Viewer3D()

def convert_to_array(s):
    tab = []
    t = s.split(' ')
    for a in t:
        if a != "":
            tab.append(a)
    return tab

def get_data_from_file(file_number):

    json_file, csv_file = False, False

    # check if json file exist
    if os.path.exists('data/object_in_'+str(file_number)+'.json'):
        # read the object and position in image form file
        json_file = True
        with open('data/object_in_'+str(file_number)+'.json', 'r') as j_f:
            obj = json.load(j_f)
    else:
        print("Not found file")

    color = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0],
         [0.117647, 0.113725, 0.392156], [1, 0.615686, 0], [0.419607, 0.317647, 0.074509]])

    # points = set()
    points = []
    pose = None
    pts, cols = [], []
    if os.path.exists('data/date-form-slam-'+ str(file_number) +'.csv'):
        # read the point and camera position from file
        csv_fiel = True
        map_state = viewer3D.Viewer3DMapElement()
        with open('data/date-form-slam-'+ str(file_number) +'.csv', 'r') as c_f:
            for l in c_f.readlines():

                temp = l.split(';')

                if len(temp[1])>3:
                    temp_pose = temp[2][1:-1].split('] [')
                    # print(temp_cure)
                    pose = np.array([convert_to_array(temp_pose[0]), convert_to_array(temp_pose[1]),
                                          convert_to_array(temp_pose[2]), convert_to_array(temp_pose[3])])
                    pose = pose.astype( np.float )
                    # print(cure_pose)
                    # map_state.cur_pose = cure_pose.astype( np.float )
                    map_state.poses.append(pose)
                    # print(len(map_state.poses))

                if len(temp[2])>3:
                    temp_cure = temp[2][1:-1].split('] [')
                    # print(temp_cure)
                    cure_pose = np.array([convert_to_array(temp_cure[0]), convert_to_array(temp_cure[1]),
                                          convert_to_array(temp_cure[2]), convert_to_array(temp_cure[3])])
                    # print(cure_pose)
                    map_state.cur_pose = cure_pose.astype( np.float )
                    map_state.poses.append(cure_pose.astype( np.float ))
                    # print(len(map_state.poses))

                if len(temp[3]) > 3:
                    # print(temp[3])
                    temp_p = temp[3][2:-3].strip().split('], [')
                    for t in temp_p:
                        po = t.split(', ')
                        id_obj = int(po[0])
                        p = np.array([float(po[1]), float(po[2]), float(po[3])])
                        points.append(p[:3])
                        map_state.colors.append(color[id_obj])
                        map_state.points.append(p)
                        if len(map_state.points) > 20000:
                            pts = np.array(map_state.points)
                            cols = np.array(map_state.colors)
                if len(pts) > 0:
                    X = np.array(map_state.points)
                    # print("{} {}".format(type(c), c.shape) )
                    Z = ward(pdist(X[:, 0::2]))
                    cluster = fcluster(Z, t=12.5, criterion='distance')
                    unique, count = np.unique(cluster, return_counts=True)
                    for n, v in zip(unique, count):
                        if v > 4:
                            res = np.where(cluster == n)
                            t = pts[res]
                            c = cols[res]
                            c_unique, c_count = np.unique(c, axis=0, return_counts=True)
                            t_xmin, t_ymin, t_zmin = min(t[:, 0]), min(t[:, 1]), min(t[:, 2])
                            t_xmax, t_ymax, t_zmax = max(t[:, 0]), max(t[:, 1]), max(t[:, 2])
                            x_size = math.fabs(t_xmax - t_xmin)
                            y_size = math.fabs(t_ymax - t_ymin)
                            z_size = math.fabs(t_zmax - t_zmin)
                            pose_x = t_xmin + (t_xmax - t_xmin)/2
                            pose_z = t_zmin + (t_zmax - t_zmin)/2
                            # map_state.box_left_botton.append([pose_x, t_ymin, pose_z])
                            # map_state.box_size.append([x_size, y_size, z_size])
                            # map_state.box_color.append(c)
                            if len(c_count) > 1:
                                c_color = c_unique[np.argmax(c_count)]
                                if str(c_color) in map_state.dict_pose:
                                    map_state.dict_pose[str(c_color)].append([[pose_x, t_ymin, pose_z], [x_size, y_size, z_size]])
                                else:
                                    map_state.dict_pose[str(c_color)] = [[[pose_x, t_ymin, pose_z], [x_size, y_size, z_size]]]
                            else:
                                if str(c_unique[0]) in map_state.dict_pose:
                                    map_state.dict_pose[str(c_unique[0])].append([[pose_x, t_ymin, pose_z], [x_size, y_size, z_size]])
                                else:
                                    map_state.dict_pose[str(c_unique[0])] = [[[pose_x, t_ymin, pose_z], [x_size, y_size, z_size]]]
                    pts, cols = [], []

                viewer.qmap.put(map_state)
            input("Press Enter to continue...")
                    # print(cure_pose)




    else:
        print("Not found file")

#


get_data_from_file("05")
