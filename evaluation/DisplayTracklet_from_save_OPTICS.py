# https://github.com/windowsub0406/KITTI_Tutorial
# https://github.com/raulmur/evaluate_ate_scale/blob/master/evaluate_ate_scale.py
import numpy as np
import glob
from src import parseTrackletXML as pt_XML
from kitti_foundation import Kitti, Kitti_util
import pandas as pd

# %matplotlib inline

box_size = 1.75

"""
recording_id = "09"
yaw_correction = -0.02
how_much_skip = 1
offset_help = 12
scale_x = scale_y = 1.57
dbscan_eps = 3#0.75#
dbscan_min = 5
min_samples = 1

MylimitX = [-50, 250]
MylimitY = [-100, 150]



recording_id = "22"
yaw_correction = -0.01
how_much_skip = 1
offset_help = 27
scale_x = scale_y = 1.82
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-200, 50]
MylimitY = [-25, 170]



recording_id = "23"
yaw_correction = 0.14
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 1.23
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-450, 50]
MylimitY = [-10, 130]



recording_id = "35"
yaw_correction = -0.02
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 1.35
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-100, 10]
MylimitY = [-30, 60]



recording_id = "39"
yaw_correction = -0.427
how_much_skip = 1
offset_help = 12
scale_x = scale_y = .9
dbscan_eps = 1
dbscan_min = 25
min_samples=1

MylimitX = [-25, 250]
MylimitY = [-10, 270]
#MylimitX = [10, 70]
#MylimitY = [15, 90]



recording_id = "46"
yaw_correction = 0.1
how_much_skip = 1
offset_help = 6
scale_x = scale_y = 0.72
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-70, 10]
MylimitY = [-70, 10]



recording_id = "61"
yaw_correction = -0.04
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 1.11
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-10, 500]
MylimitY = [-20, 80]



recording_id = "64"
yaw_correction = 0.03
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 0.635
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-50, 320]
MylimitY = [-10, 400]



recording_id = "84"
yaw_correction = 0.02
how_much_skip = 1
offset_help = 75
scale_x = scale_y = 1.93
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-50, 150]
MylimitY = [-10, 280]


"""
recording_id = "91"
yaw_correction = -0.01
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 0.83
dbscan_eps = 1.
dbscan_min = 25
min_samples=1

MylimitX = [-250, 25]
MylimitY = [-30, 10]


SaveFig = True

draw_path1 = False
draw_path2 = False

draw_points2 = False
draw_rect1 = False
draw_rect2 = True

#TO NIE :-)
#Można usunąć
"""
recording_id = "93"
yaw_correction = .0
how_much_skip = 1
offset_help = 1
scale_x = scale_y = 0.83
dbscan_eps = 1
dbscan_min = 5
min_samples=1

"""



if draw_rect1 == True:
    fig_name = "results/optics_reference_id=" + recording_id + ".png"
if draw_rect2 == True:
    #fig_name = "results/optics_results_id=" + recording_id + ",eps=" + str(dbscan_eps) + ",min=" + str(dbscan_min) + ",box_size=" + str(box_size) + ".png"
    fig_name = "results/optics_results_id=" + recording_id + ",min=" + str(dbscan_min) + ",box_size=" + str(box_size) + ".png"
if draw_points2 == True:
    fig_name = "results/optics_results_id=" + recording_id + ",eps=" + str(dbscan_eps) + ",min=" + str(dbscan_min) + ",points_dilation.png"
"""
draw_rect1 = False
draw_rect2 = True
"""




listX, listY = [], []

from shapely.geometry import Polygon
Polygons1 = []
Polygons2 = []

image_type = 'color'  # 'gray' or 'color' image
mode = '00' if image_type == 'gray' else '02'  # image_00 = 'graye image' , image_02 = 'color image'

path_to_data = 'd:\\dane\\kitti\\2011_09_26\\'
subfolder = '2011_09_26_drive_00' + recording_id + '_sync\\'

image_path = path_to_data + subfolder + 'image_' + mode + '/data'
velo_path = path_to_data + subfolder + '/velodyne_points/data'
xml_path = path_to_data + subfolder + "/tracklet_labels.xml"
v2c_filepath = path_to_data + '/calib_velo_to_cam.txt'
c2c_filepath = path_to_data + '/calib_cam_to_cam.txt'
frame = 40



#file_to_save_tracklets = 'tracklets22.txt'
#file_to_save_tracklets = 'tracklets39.txt'
file_to_save_tracklets = 'tracklets' + recording_id + '.txt'

#df2 = pd.read_csv("df22.txt", sep=',').to_numpy()
#df2 = pd.read_csv("../df39.txt", sep=',').to_numpy()
df2 = pd.read_csv("df" + recording_id + ".txt", sep=',').to_numpy()

yaw = df2[:,5]
yaw_help = df2[0,5] + yaw_correction
#yaw_help = df2[offset_help,5]

#frame='all'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111)
#plt.title("3D Tracklet display")

#pnt = points.T[:, 1::5]  # one point in 5 points

#df = pd.read_csv("d:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_0039_sync\\ground_truth.txt")
#df = pd.read_csv("data22.txt", header=None, sep=' ')
#df = pd.read_csv("data39.txt", header=None, sep=' ')
df = pd.read_csv("data" + recording_id + ".txt", header=None, sep=' ')
df_tracklets = pd.read_csv(file_to_save_tracklets, header=None, sep=',')

df_numpy = df.to_numpy()

df_tracklets_numpy = df_tracklets.to_numpy()


pp = np.zeros((df_numpy.shape[0], 3))

pp[:,0] = df_numpy[:,0]
pp[:,1] = df_numpy[:,1]

trans_error = 0
reference_data = 0



#TUTUTUTU
if draw_path1:
    ax.scatter(x = df_numpy[:,0], y = df_numpy[:,1], s=5, c='k', marker='.')#, alpha=0.5)
reference_data = np.copy(np.array([df_numpy[:,0], df_numpy[:,1]]))



#my_data_help = np.zeros(1 + (3 * 8) + 1)

def draw_3d_box(tracklet_, df_numpy):
    """ draw 3d bounding box """

    type_c = {'Car': 'b', 'Van': 'g', 'Truck': 'r', 'Pedestrian': 'c', \
              'Person (sitting)': 'm', 'Cyclist': 'y', 'Tram': 'k', 'Misc': 'w'}

    type_int = {'0': 'b', '1': 'g', '2': 'r', '3': 'c', \
              '4': 'm', '5': 'y', '6': 'k', '7': 'w'}


    line_order = ([0, 1], [1, 2], [2, 3], [3, 0])
    xx = np.zeros(3)
    xx[0] = df_numpy[0]
    xx[1] = df_numpy[1]





    tt = np.zeros((8,3))
    for a in range(tt.shape[0]):
        for b in range(tt.shape[1]):
            tt[a][b] = tracklet_[1 + ((a * 3) + b)]

    my_key = str(int(tracklet_[1 + 24]))

    for i in range(tt.shape[0]):
        for k in line_order:
            if draw_rect1:
                #ax.plot(*zip(tt[k[1]] + xx, tt[k[0]] + xx), lw=1.5, color=type_int[my_key])
                ax.plot(*zip(tt[k[1]] + xx, tt[k[0]] + xx), lw=1.5, color='b')

    pol = Polygon([(tracklet_[1],tracklet_[2]),
                   (tracklet_[4],tracklet_[5]),
                   (tracklet_[7],tracklet_[8]),
                   (tracklet_[10],tracklet_[11])])
    Polygons1 .append(pol)


for frame in range(yaw.shape[0]):
#for frame in range(1):

    #check = Kitti_util(frame=frame, velo_path=velo_path, camera_path=image_path, \
    #                   xml_path=xml_path, v2c_path=v2c_filepath, c2c_path=c2c_filepath, yaw_correction = yaw[frame])

    # bring velo points & tracklet info
    #points = check.velo_file
    #tracklet_, type_ = check.tracklet_info
    end_loop = False
    dd = 0
    found = -1

    while not(end_loop):
        if dd >= df_tracklets_numpy.shape[0]:
            end_loop = True
        else:
            if df_tracklets_numpy[dd][0] == frame:
                found = dd
            #end_loop = True
        #else:
        #    dd = dd + 1


        if found > -1:
            draw_3d_box(df_tracklets_numpy[dd], df_numpy[frame])
            found = -1
        dd = dd + 1

ax.set_xlabel('X')
ax.set_ylabel('Y')


########################################
########################################

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("data" + recording_id + ".txt", header=None, sep=' ')

df_numpy = df.to_numpy()
pp = np.zeros((df_numpy.shape[0], 3))
pp[:, 0] = np.copy(df_numpy[:, 0])
pp[:, 1] = np.copy(df_numpy[:, 1])


yaw = 3.14 / 2 - yaw_help


def convert_to_array(s):
    # usuwanie pustych elementów z tablicy i zwracanie rekordów macierzy
    tab = []
    t = s.split(' ')
    for a in t:
        if a != "":
            tab.append(float(a))
    return tab

import math
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    #R = R.T
    #assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def show_2D_map(file_number):
    # kolor przypisany dla odpowiednich obiektów

    # definicja kolorów dla obiektów format RGB
    # Car - [1, 0, 0]   Van - [0, 1, 0]     Truck - [0, 0, 1]   Person - [1, 0, 1]
    # Bus - [0, 1, 1]   Cyclist - [1, 1, 0] Tram  - [0.117647, 0.113725, 0.392156]
    # Traffic light - [1, 0.615686, 0],
    # Traffic sign - [0.419607, 0.317647, 0.074509]

    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0],
             [0.117647, 0.113725, 0.392156], [1, 0.615686, 0], [0.419607, 0.317647, 0.074509]]

    points = {}
    pose_camer_x, pose_camer_y = [], []
    pp_camera_x, pp_camera_y = [], []
    with open('data/date-form-slam-' + str(file_number) + '.csv', 'r') as c_f:
        for l in c_f.readlines():
            temp = l.split(';')
            if len(temp[2]) > 3:
                # pozyskanie macierzy aktualnej pozycji kamery (4x4)
                temp_pose = temp[2][1:-2].split('] [')
                pose = np.array([convert_to_array(temp_pose[0]), convert_to_array(temp_pose[1]),
                                 convert_to_array(temp_pose[2]), convert_to_array(temp_pose[3])])
                # pobieranie pozycji kamery (x, z) z ostatniej kolumny macierzy (4 x 4)
                pose_camer_x.append(pose[0, 3])
                pose_camer_y.append(pose[2, 3])


    item_counter = 0
    with open('data/point-from-slam-' + str(file_number) + '.csv', 'r') as fr:
        for pl in fr.readlines():
            if item_counter % how_much_skip == 0:
                if len(pl) > 3:
                    # dzielenie w celu uzsykania koluru punktu oraz wzpółrzędnych x,y,z
                    temp_p = pl.strip().split(';')
                    id_obj = int(temp_p[0])  # id obiektu z YOLO
                    if id_obj in points:
                        points[id_obj].append([float(temp_p[1]), float(temp_p[3])])  # dodawanie punktów x, z
                    else:
                        points[id_obj] = [[float(temp_p[1]), float(temp_p[3])]]  # dodawanie punktów x, z
            item_counter = item_counter + 1
    # rysowanie drogi

    mm = np.zeros((3, np.array(pose_camer_x).shape[0]))
    mm[1, :] = np.array(pose_camer_x)
    mm[0, :] = np.array(pose_camer_y)

    rotMat = np.array([ \
        [np.cos(yaw), -np.sin(yaw), 0.0], \
        [np.sin(yaw), np.cos(yaw), 0.0], \
        [0.0, 0.0, 1.0]])


    mm2 = np.matmul(rotMat, mm)


    # cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

    # pose_camer_x = np.array(pose_camer_x) + pp[offset_help,0]
    # pose_camer_y = np.array(pose_camer_y) + pp[offset_help,1]


    pose_camer_x = mm2[1, :] * scale_x
    pose_camer_y = mm2[0, :] * scale_y

    xx_xx = pose_camer_x[0]
    yy_yy = pose_camer_y[0]

    pose_camer_x = pose_camer_x + pp[offset_help, 0] - xx_xx
    pose_camer_y = pose_camer_y + pp[offset_help, 1] - yy_yy

    #plt.plot(pose_camer_x, pose_camer_y, "g-")
    if draw_path2:
        ax.scatter(x=pose_camer_x, y=pose_camer_y, s=5, c='r', marker='.')  # , alpha=0.5)
    global results_data
    results_data = np.copy(np.array([pose_camer_x, pose_camer_y]))


    rotMat = np.array([ \
        [np.cos(-yaw), -np.sin(-yaw), 0.0], \
        [np.sin(-yaw), np.cos(-yaw), 0.0], \
        [0.0, 0.0, 1.0]])
    # rysowanie punktów
    for key, point_value in points.items():
        # pobieranie odpowiedniego koloru dla grupy punktów
        color_p = color[key]
        for p in point_value:
            mm = np.zeros((3, 1))
            mm[0, 0] = p[0]
            mm[1, 0] = p[1]

            mm2 = np.matmul(rotMat, mm)
            #mm2[0, :] = mm2[0, :] + pp[offset_help, 0]
            #mm2[1, :] = mm2[1, :] + pp[offset_help, 1]
            p[0] = mm2[0, 0] * scale_x
            p[1] = mm2[1, 0] * scale_y

            p[0] = p[0] + pp[offset_help, 0] - xx_xx
            p[1] = p[1] + pp[offset_help, 1] - yy_yy

            listX.append(p[0])
            listY.append(p[1])
            #if draw_points2:
            #    ax.scatter(x=p[0], y=p[1], s=5, c="green", marker='.')  # , alpha=0.5)

# w wyłoaniu funkcji wystarczy podać numer plików z których korzystamy
# cała naza pliku definiweana jest wewnąrzt funkcji

# show_2D_map("64")
#show_2D_map("22")
#show_2D_map("39")
show_2D_map(recording_id)

########################################
########################################

reference_data = reference_data[:,offset_help:reference_data.shape[1]]
trans_min = min(reference_data.shape[1], results_data.shape[1])
err = 0
ang_error = 0

def angle_between(vector_1, vector_2):
    #vector_1 = [0, 1]
    #vector_2 = [1, 0]
    unit_vector_1 = vector_1 / np. linalg. norm(vector_1)
    unit_vector_2 = vector_2 / np. linalg. norm(vector_2)
    dot_product = np. dot(unit_vector_1, unit_vector_2)
    if dot_product >= 1:
        dot_product = 1
    if dot_product <= -1:
        dot_product = -1
    angle = np.arccos(dot_product)
    if math.isnan(angle):
        aaaaa = 1
        aaaaa = aaaaa + 1
    return angle


for a in range(trans_min):

    xx = reference_data[0, a]
    yy = reference_data[1, a]

    xx1 = results_data[0, a]
    yy1 = results_data[1, a]

    _len = np.sqrt(xx * xx + yy * yy)
    #err = err + (((xx - xx1) * (xx - xx1)) + ((yy - yy1) * (yy - yy1)) / _len)
    err = err + (math.sqrt(((xx - xx1) * (xx - xx1)) + ((yy - yy1) * (yy - yy1))) / _len)

    angle = angle_between([xx, yy], [xx1, yy1])
    angle = angle / _len
    ang_error = ang_error + angle
    #print(angle)

print(err / trans_min * 100)
print((ang_error / trans_min)/np.pi * 180)

vv = np.asarray([listX, listY]).T
#np.save("points", vv)
#print(vv.shape)

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
#clustering = DBSCAN(eps=0.75, min_samples=5).fit(vv)
clustering = OPTICS(#eps=dbscan_eps,
                    min_samples=dbscan_min).fit(vv)
#clustering = DBSCAN(eps=1, min_samples=3).fit(vv)
vvc = np.copy(clustering.labels_) + 1

if draw_points2:
    for id_help in range(vv.shape[0]):
        if vvc[id_help] > 1:
            ax.scatter(x=vv[id_help,0], y=vv[id_help,1], s=1, c="green", marker='.')  # , alpha=0.5)


#np.save("points_cluster", vvc)

vvv_unique = np.unique(vvc)
my_colors = []
bbbbbb = len(vvv_unique)


cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(vvv_unique))]
import random
random.shuffle(colors)

ccc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



ind = [[0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 1, 2],

       [1, 0, 2, 3],
       [1, 0, 3, 2],
       [1, 2, 0, 3],
       [1, 2, 3, 0],
       [1, 3, 0, 2],
       [1, 3, 2, 0],

       [2, 0, 1, 3],
       [2, 0, 3, 1],
       [2, 1, 0, 3],
       [2, 1, 3, 0],
       [2, 3, 0, 1],
       [2, 3, 1, 0],

       [3, 0, 1, 2],
       [3, 0, 2, 1],
       [3, 1, 0, 2],
       [3, 1, 2, 0],
       [3, 2, 0, 1],
       [3, 2, 1, 0]
       ]


from sklearn.decomposition import PCA
def draw_rect(xxx, ax, my_col):
    pca = PCA(n_components=2)
    pca.fit(xxx)
    #print(pca.components_)
    v_my = []
    v_v = []
    for length, vector in zip(pca.explained_variance_, pca.components_):
        #v = vector * 2.5 * np.sqrt(length)
        v = vector * box_size * np.sqrt(length)
        #print(pca.components_)
        #ax.plot((198, 210), (-49, -43))
        #ax.plot((pca.mean_[0] - v[0], pca.mean_[0] + v[0]),(pca.mean_[1] - v[1], pca.mean_[1] + v[1]))
        v_my.append((pca.mean_[0] - v[0], pca.mean_[0] + v[0],pca.mean_[1] - v[1], pca.mean_[1] + v[1]))
        v_v.append(v)
        #draw_vector(pca.mean_, pca.mean_ + v)

    #ax.plot((v_my[0][0], v_my[0][1]),(v_my[0][2], v_my[0][3]))
    #ax.plot((v_my[1][0], v_my[1][1]),(v_my[1][2], v_my[1][3]))

    #ax.plot((v_my[1][0] - v_v[1][0], v_my[1][1] - v_v[1][0]),(v_my[1][2], v_my[1][3]))

    if draw_rect2:
        ax.plot((pca.mean_[0] - v_v[0][0] - v_v[1][0], pca.mean_[0] + v_v[0][0] - v_v[1][0]),(pca.mean_[1] - v_v[0][1] - v_v[1][1], pca.mean_[1] + v_v[0][1]  - v_v[1][1]), color=my_col)
        ax.plot((pca.mean_[0] - v_v[0][0] + v_v[1][0], pca.mean_[0] + v_v[0][0] + v_v[1][0]),(pca.mean_[1] - v_v[0][1] + v_v[1][1], pca.mean_[1] + v_v[0][1]  + v_v[1][1]), color=my_col)


        #ax.plot((pca.mean_[0] - v_v[1][0] - v_v[0][0], pca.mean_[0] + v_v[1][0] + v_v[0][0]),(pca.mean_[1] - v_v[1][1] - v_v[0][0], pca.mean_[1] + v_v[1][1] - v_v[0][0]))
        ax.plot((pca.mean_[0] - v_v[1][0] - v_v[0][0], pca.mean_[0] + v_v[1][0] - v_v[0][0]),(pca.mean_[1] - v_v[1][1] - v_v[0][1], pca.mean_[1] + v_v[1][1] - v_v[0][1]), color=my_col)
        ax.plot((pca.mean_[0] - v_v[1][0] + v_v[0][0], pca.mean_[0] + v_v[1][0] + v_v[0][0]),(pca.mean_[1] - v_v[1][1] + v_v[0][1], pca.mean_[1] + v_v[1][1] + v_v[0][1]), color=my_col)

    vvv = [
        (pca.mean_[0] - v_v[0][0] - v_v[1][0], pca.mean_[0] + v_v[0][0] - v_v[1][0]),
        (pca.mean_[1] - v_v[0][1] - v_v[1][1], pca.mean_[1] + v_v[0][1] - v_v[1][1]),
        (pca.mean_[0] - v_v[0][0] + v_v[1][0], pca.mean_[0] + v_v[0][0] + v_v[1][0]),
        (pca.mean_[1] - v_v[0][1] + v_v[1][1], pca.mean_[1] + v_v[0][1] + v_v[1][1])
    ]
    #
    #ind = [0,1,2,3]

    #ind = [3, 2, 0, 1]
    aaa = 0
    ind_h = ind[aaa]
    pol = Polygon([
        vvv[ind_h[0]],
        vvv[ind_h[1]],
        vvv[ind_h[2]],
        vvv[ind_h[3]]
    ])

    while not(pol.is_valid) and aaa < len(ind) -1:
        aaa = aaa + 1
        ind_h = ind[aaa]
        pol = Polygon([
            vvv[ind_h[0]],
            vvv[ind_h[1]],
            vvv[ind_h[2]],
            vvv[ind_h[3]]
        ])

    Polygons2.append(pol)

for a in range(1, len(vvv_unique)):
    xxx = vv[vvc == a,]
    draw_rect(xxx, ax, 'green')


import pickle
outfile = open("Polygons1.txt",'wb')
pickle.dump(Polygons1,outfile, protocol=pickle.HIGHEST_PROTOCOL)
outfile.close()

outfile = open("Polygons2.txt",'wb')
pickle.dump(Polygons2,outfile, protocol=pickle.HIGHEST_PROTOCOL)
outfile.close()

print(Polygons1[0])

#https://stackoverflow.com/questions/57885406/get-the-coordinates-of-two-polygons-intersection-area-in-python?rq=1

if SaveFig:
    plt.xlim(MylimitX)
    plt.ylim(MylimitY)
    plt.axis('off')
else:
    #plt.xlim(MylimitX)
    #plt.ylim(MylimitY)

    #fig.savefig(fig_name + ".eps")
    plt.show()


if SaveFig:
    fig.savefig(fig_name)
    import cv2
    img = cv2.imread(fig_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    img[img > 0] = 255

    if draw_rect2:
        #################
        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255);
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = img | im_floodfill_inv
    else:
        im_floodfill = img.copy()
        #kernel = np.ones((3, 3), np.uint8)
        #im_out = cv2.dilate(im_floodfill, kernel, iterations=1)
        im_out = img


    cv2.imshow("a",im_out)
    cv2.imwrite(fig_name, im_out)
    cv2.waitKey()
#for p1 in Polygons2:
#    print(p1.is_valid)

#https://github.com/raulmur/evaluate_ate_scale/blob/master/associate.py
#https://github.com/raulmur/evaluate_ate_scale/blob/master/evaluate_ate_scale.py
#https://github.com/YuePanEdward/MULLS/blob/2a828f8961d65e814809d62cdd0dad5109c05ec6/python/kitti_eval.py#L166
#https://github.com/YuePanEdward/MULLS/blob/2a828f8961d65e814809d62cdd0dad5109c05ec6/python/kitti_eval.py#L152
#https://github.com/YuePanEdward/MULLS/blob/main/python/diagnostics.py
#https://github.com/YuePanEdward/MULLS
#http://www.cvlibs.net/datasets/kitti/eval_odometry.php
