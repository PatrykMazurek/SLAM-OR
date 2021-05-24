#https://github.com/windowsub0406/KITTI_Tutorial
import numpy as np
import glob
from src import parseTrackletXML as pt_XML
from kitti_foundation import Kitti, Kitti_util
import pandas as pd
#%matplotlib inline

image_type = 'color'  # 'gray' or 'color' image
mode = '00' if image_type == 'gray' else '02'  # image_00 = 'graye image' , image_02 = 'color image'

file_id = "93"

path_to_data = 'd:\\dane\\kitti\\2011_09_26\\'
subfolder = '2011_09_26_drive_00' + file_id + '_sync\\'

image_path = path_to_data + subfolder + 'image_' + mode + '/data'
velo_path = path_to_data + subfolder + '/velodyne_points/data'
xml_path = path_to_data + subfolder + "/tracklet_labels.xml"
v2c_filepath = path_to_data + '/calib_velo_to_cam.txt'
c2c_filepath = path_to_data + '/calib_cam_to_cam.txt'
frame = 40

file_to_save_tracklets = 'tracklets'+ file_id + '.txt'

df2 = pd.read_csv("../df" + file_id + ".txt", sep=',').to_numpy()

yaw = df2[:,5]

#frame='all'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection='3d')
plt.title("3D Tracklet display")
#pnt = points.T[:, 1::5]  # one point in 5 points

#df = pd.read_csv("d:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_0039_sync\\ground_truth.txt")
df = pd.read_csv("data" + file_id + ".txt", header=None, sep=' ')


df_numpy = df.to_numpy()
print(df_numpy.shape)
pp = np.zeros((df_numpy.shape[0], 3))
print(pp.shape)
pp[:,0] = df_numpy[:,0]
pp[:,1] = df_numpy[:,1]
#pp = np.transpose(pp)
#print(pp)
#print(pnt)
print(df_numpy[:,0].shape)
ax.scatter(xs = df_numpy[:,0], ys = df_numpy[:,1], zs = df_numpy[:,2], s=5, c='k', marker='.')#, alpha=0.5)
#print(pnt)
#ax.scatter(*pnt, s=0.1, c='k', marker='.', alpha=0.5)

#print(tracklet_)

#print(tracklet_[0])

my_data_help = np.zeros(1 + (3 * 8) + 1)

def draw_3d_box(tracklet_, type, df_numpy):
    """ draw 3d bounding box """

    type_c = {'Car': 'b', 'Van': 'g', 'Truck': 'r', 'Pedestrian': 'c', \
              'Person (sitting)': 'm', 'Cyclist': 'y', 'Tram': 'k', 'Misc': 'w'}

    type_int = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, \
              'Person (sitting)': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], \
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    xx = np.zeros(3)
    xx[0] = df_numpy[0]
    xx[1] = df_numpy[1]
    xx[2] = df_numpy[2]
    #print(xx)

    for i, j in zip(tracklet_[frame], type_[frame]):
        print(i.T)
        for a in range(8):
            for b in range(3):
                my_data_help[1 + ((a * 3) + b)] = i.T[a,b]
        my_data_help[1 + 24] = type_int[j]

        if type_int[j] == 2:
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')

        file_object = open(file_to_save_tracklets, 'a')
        for a in range(my_data_help.shape[0]):
            if a > 0:
                file_object.write(',')
            file_object.write(str(my_data_help[a]))
        file_object.write('\n')
        file_object.close()

        for k in line_order:
            #print(i.T[k[1]])
            #ax.plot3D(*zip(i.T[k[1]], i.T[k[0]]), lw=1.5, color=type_c[j])
            #ax.plot3D(*zip(i.T[k[1]] + df_numpy, i.T[k[0]] + df_numpy), lw=1.5, color=type_c[j])
            ax.plot3D(*zip(i.T[k[1]] + xx, i.T[k[0]] + xx), lw=1.5, color=type_c[j])


for frame in range(yaw.shape[0]):

    my_data_help[0] = frame

    check = Kitti_util(frame=frame, velo_path=velo_path, camera_path=image_path, \
                       xml_path=xml_path, v2c_path=v2c_filepath, c2c_path=c2c_filepath, yaw_correction = yaw[frame])

    # bring velo points & tracklet info
    points = check.velo_file
    tracklet_, type_ = check.tracklet_info

    print(check)
    #print(points.shape)
    if tracklet_[frame] is not None:
        print('The number of GT : ', len(tracklet_[frame]))

        from mpl_toolkits.mplot3d import Axes3D

        draw_3d_box(tracklet_, type_, df_numpy[frame])


#print(df_numpy[frame,:])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#ax.set_xlim3d(-10, 260)
#ax.set_ylim3d(-20, 240)
#ax.set_zlim3d(-2, 15)

plt.show()

