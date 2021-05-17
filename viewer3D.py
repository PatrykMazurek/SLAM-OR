"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import config
import multiprocessing as mp
import math, time, os

from multiprocessing import Process, Queue, Value
import pangolin
import OpenGL.GL as gl
import numpy as np
import frame
from utils_geom import inv_T

# from scipy.cluster.hierarchy import fclusterdata

from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

kUiWidth = 180
kDefaultPointSize = 2
kViewportWidth = 1024
#kViewportHeight = 768
kViewportHeight = 550
kDrawCameraPrediction = False   
kDrawReferenceCamera = True   

kMinWeightForDrawingCovisibilityEdge=100


class Viewer3DMapElement(object): 
    def __init__(self):
        self.cur_pose = None 
        self.predicted_pose = None 
        self.reference_pose = None 
        self.poses = [] 
        self.points = [] 
        self.colors = []         
        self.covisibility_graph = []
        self.spanning_tree = []        
        self.loops = []
        # dodatkowa tabela dla boxa
        self.box_left_botton = []   # min (x, y, z)
        self.box_size = []     # max (x, y, z)
        self.box_color = []
        self.dict_pose = {}     # zawiera pozycje obiektu, rozmiar
        self.line = []          # linia zaznaczająca pozycję kamery
        
              
class Viewer3DVoElement(object): 
    def __init__(self):
        self.poses = [] 
        self.traj3d_est = []   # estimated trajectory 
        self.traj3d_gt = []    # ground truth trajectory            
        

class Viewer3D(object):
    def __init__(self):
        self.map_state = None
        self.qmap = Queue()
        self.vo_state = None
        self.qvo = Queue()        
        self._is_running  = Value('i',1)
        self._is_paused = Value('i',1)
        self.vp = Process(target=self.viewer_thread,
                          args=(self.qmap, self.qvo,self._is_running ,self._is_paused,))
        self.vp.daemon = True
        self.vp.start()

        self.kps = None

    def quit(self):
        self._is_running.value = 0
        self.vp.join()
        #pangolin.Quit()
        print('Viewer stopped')   
        
    def is_paused(self):
        return (self._is_paused.value == 1)

    def viewer_thread(self, qmap, qvo, is_running, is_paused):
        self.viewer_init(kViewportWidth, kViewportHeight)
        while not pangolin.ShouldQuit() and (is_running.value == 1):
            self.viewer_refresh(qmap, qvo, is_paused)
        print('Quitting viewer...')    

    def viewer_init(self, w, h):
        # pangolin.ParseVarsFile('app.cfg')

        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x = 0
        viewpoint_y = -40
        viewpoint_z = -80
        viewpoint_f = 1000
            
        self.proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w//2, h//2, 0.1, 5000)
        self.look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, kUiWidth/w, 1.0, -w/h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.0, 1.0, 0.0, kUiWidth/w)

        self.do_follow = True
        self.is_following = True 
        
        self.draw_cameras = True
        self.draw_covisibility = True        
        self.draw_spanning_tree = True           
        self.draw_loops = True                

        #self.button = pangolin.VarBool('ui.Button', value=False, toggle=False)
        self.checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
        self.checkboxCams = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
        self.checkboxCovisibility = pangolin.VarBool('ui.Draw Covisibility', value=True, toggle=True)  
        self.checkboxSpanningTree = pangolin.VarBool('ui.Draw Tree', value=True, toggle=True)                
        self.checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)           
        self.checkboxPause = pangolin.VarBool('ui.Pause', value=False, toggle=True)             
        #self.float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
        #self.float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
        self.int_slider = pangolin.VarInt('ui.Point Size', value=kDefaultPointSize, min=1, max=10)  

        self.pointSize = self.int_slider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        # print("self.Twc.m",self.Twc.m)


    def viewer_refresh(self, qmap, qvo, is_paused):

        while not qmap.empty():
            self.map_state = qmap.get()

        while not qvo.empty():
            self.vo_state = qvo.get()

        # if pangolin.Pushed(self.button):
        #    print('You Pushed a button!')

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()        
        self.draw_cameras = self.checkboxCams.Get()
        self.draw_covisibility = self.checkboxCovisibility.Get()
        self.draw_spanning_tree = self.checkboxSpanningTree.Get()
        
        #if pangolin.Pushed(self.checkboxPause):
        if self.checkboxPause.Get():
            is_paused.value = 0  
        else:
            is_paused.value = 1  
                    
        # self.int_slider.SetVal(int(self.float_slider))
        self.pointSize = self.int_slider.Get()
            
        if self.do_follow and self.is_following:
            self.scam.Follow(self.Twc, True)
        elif self.do_follow and not self.is_following:
            self.scam.SetModelViewMatrix(self.look_view)
            self.scam.Follow(self.Twc, True)
            self.is_following = True
        elif not self.do_follow and self.is_following:
            self.is_following = False            

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        
        self.dcam.Activate(self.scam)

        if self.is_grid:
            Viewer3D.drawPlane()

        # ==============================
        # draw map 
        if self.map_state is not None:
            if self.map_state.cur_pose is not None:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.cur_pose)
                gl.glLineWidth(1)                
                self.updateTwc(self.map_state.cur_pose)
                
            if self.map_state.predicted_pose is not None and kDrawCameraPrediction:
                # draw predicted pose in red
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCamera(self.map_state.predicted_pose)           
                
            if len(self.map_state.poses) >1:
                # draw keyframe poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.map_state.poses[:])

            # rysowaine wektora za którym powinno wyznaczać boxy
            # if len(self.map_state.line) > 0 :
            # #     # wyznaczam nowy wektor obrócony o -90 i 90 stopni o długości 10
            #     gl.glLineWidth(3)
            #     gl.glPointSize(self.pointSize)
            #     gl.glColor3f(1.0, 0.0, 1.0)
            #     pangolin.DrawPoints(self.map_state.line)
            #     # pangolin.DrawLines([self.map_state.line[0], self.map_state.line[0]])


            if len(self.map_state.points)>0:
                # draw keypoints with their color
                gl.glPointSize(self.pointSize)
                #gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.map_state.points, self.map_state.colors)

            # TODO dopisać rozwiązanie drukujące na ekran zaznaczone obiekty w postaci obszarów
            if len(self.map_state.dict_pose) > 0:
                for k, v in self.map_state.dict_pose.items():
                    sizes = []
                    poses = [ np.identity(4) for i in range(len(v))]
                    for pose, point in zip(poses, v):
                        pose[:3, 3] = np.array(point[0])
                        sizes.append( np.array(point[1]) )
                    gl.glLineWidth(2)
                    d_color = k[1:-1].split(' ')
                    gl.glColor3f(float(d_color[0]), float(d_color[1]), float(d_color[2]))
                        # gl.glColor3f(1.0, 0.0, 1.0)
                # gl.glColor3f(self.map_state.box_color)
                #     print("{} {}".format(poses, sizes))
                    pangolin.DrawBoxes(poses, sizes)
                # pangolin.DrawBoxes(self.map_state.box_left_botton, self.map_state.box_size)

            if self.map_state.reference_pose is not None and kDrawReferenceCamera:
                # draw predicted pose in purple
                gl.glColor3f(0.5, 0.0, 0.5)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.reference_pose)      
                gl.glLineWidth(1)          
                
            if len(self.map_state.covisibility_graph)>0:
                if self.draw_covisibility:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLines(self.map_state.covisibility_graph,3)                                             
                    
            if len(self.map_state.spanning_tree)>0:
                if self.draw_spanning_tree:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 0.0, 1.0)
                    pangolin.DrawLines(self.map_state.spanning_tree,3)              
                    
            if len(self.map_state.loops)>0:
                if self.draw_spanning_tree:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.5, 0.0, 0.5)
                    pangolin.DrawLines(self.map_state.loops,3)        
                    gl.glLineWidth(1)                                               

        # ==============================
        # draw vo 
        if self.vo_state is not None:
            if self.vo_state.poses.shape[0] >= 2:
                # draw poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.vo_state.poses[:-1])

            if self.vo_state.poses.shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                current_pose = self.vo_state.poses[-1:]
                pangolin.DrawCameras(current_pose)
                self.updateTwc(current_pose[0])

            if self.vo_state.traj3d_est.shape[0] != 0:
                # draw blue estimated trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.vo_state.traj3d_est)

            if self.vo_state.traj3d_gt.shape[0] != 0:
                # draw red ground-truth trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawLine(self.vo_state.traj3d_gt)                


        pangolin.FinishFrame()


    def draw_map(self, slam, list_box, frame_id, file_name, save_data):
        if self.qmap is None:
            return
        map = slam.map
        map_state = Viewer3DMapElement()
        color = np.flip([0., 0., 0.])
        temp_p, temp_c = [], []
        point_after_camera, point_color_after_camera = [], []

        if map.num_frames() > 0:
            map_state.cur_pose = map.get_frame(-1).Twc.copy()
            if map.num_frames() > 1:
                prev_pose = map.get_frame(-2).Twc.copy()
                cure_pose = map_state.cur_pose.copy()

                p_r, p_t = prev_pose[:3,:3], prev_pose[:, 3]
                c_r, c_t = cure_pose[:3,:3], cure_pose[:, 3]

                p_p = p_r.dot(p_t[:3])
                c_p = c_r.dot(c_t[:3])

                n_p = c_p - p_p
                print("c_p {}".format(c_p))

                # vector_length = math.sqrt( (cure_pose[0] - prev_pose[0])**2
                #                            + (cure_pose[1] - prev_pose[1])**2
                #                            + (cure_pose[2] - prev_pose[2])**2)

                x, y, z = np.radians(0), np.radians(0), np.radians(90)

                # x
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(x), -np.sin(x)],
                               [0, np.sin(x), np.cos(x)],
                               ], dtype=np.float)
                # y
                Ry = np.array([[np.cos(y), 0, np.sin(y)],
                               [0, 1, 0],
                               [-np.sin(y), 0, np.cos(y)],
                                ], dtype=np.float)
                # z
                Rz = np.array([[np.cos(z), -np.sin(z), 0],
                               [np.sin(z), np.cos(z), 0],
                               [0, 0, 1],
                                ], dtype=np.float)

                R = Rz.dot(Ry.dot(Rx))


                # print("vector lenght {}".format(vector_length))
                temp_tab = R.dot(c_p)
                print(temp_tab)
                pot = [[temp_tab[0] + n_p[0], temp_tab[1], temp_tab[2] + n_p[2]],
                        [temp_tab[0], temp_tab[1], temp_tab[2]],
                        [temp_tab[0], temp_tab[1], temp_tab[2]]]

                # new_point = np.array([temp_tab[0] *3, temp_tab[1], temp_tab[2] * 3])
                # # # print("new point {}".format(new_point))
                # # # line_f[1] = line_f[1] * (vector_length * 3)
                # # print("c {}".format(cure_pose))
                # # print("p {}".format(new_point))
                map_state.line = pot


        if slam.tracking.predicted_pose is not None: 
            map_state.predicted_pose = slam.tracking.predicted_pose.inverse().matrix().copy()


        if slam.tracking.kf_ref is not None: 
            reference_pose = slam.tracking.kf_ref.Twc.copy()

        num_map_keyframes = map.num_keyframes()
        keyframes = map.get_keyframes()
        # print(map_state.predicted_pose)
        if num_map_keyframes>0:
            for kf in keyframes:
                map_state.poses.append(kf.Twc)  
        map_state.poses = np.array(map_state.poses)

        # przypisanie punktów do widoku

        #print("punkty z obiektu map {}".format(num_map_points))
        # list_test_point = []
        # zaznaczenie punktów 3D na mapie, ounkty reprezentują odnaleziony obiekt
        # todo sprawdzić czy jest przemieszczenie

        if num_map_keyframes > 0:
            # odwołanie się do obiektu KeyFrame
            temp_frame = map.get_last_keyframe()
            ids_kps = len(temp_frame.kps)
            temp_point_set = list()
            # sprawdzenie długości listy kps
            nr = 0
            for id in range(0, ids_kps):
                map_point = temp_frame.points[id]
                if map_point is not None and not map_point.is_bad:
                    uv = tuple(temp_frame.kps[id])
                    for box in list_box:
                        if box.xmin < uv[0] and box.xmax > uv[0] and box.ymin < uv[1] and box.ymax > uv[1]:
                            map_point.color = np.flip(box.color)
                            # dodaj do obiektu wspórzędne punktu
                            # box.pose.append(map_point.pt)
                            # list_test_point.append(map_point)
                            # temp_point_set.add(map_point)
                            map.add_map_point(map_point)
            print("Dodane punkty {}".format(nr))

        # zapisanie danych do pliku csv w kolejności nr_klatki ; pose ; cur_pose ; points [id_class, x, y, z] ;
        temp_cure_pose = ""
        temp_poses = ""
        color = np.array([np.flip([1, 0, 0]), np.flip([0, 1, 0]), np.flip([0, 0, 1]), np.flip([1, 0, 1]), np.flip([0, 1, 1]),
                 np.flip([1, 1, 0]), np.flip([0.117647, 0.113725, 0.392156]), np.flip([1, 0.615686, 0])
                 ,np.flip([0.419607, 0.317647, 0.074509])])
        list_points_c = []

        # pruba rotacji wektora



        if save_data:
            with open("date-form-slam-"+ str(file_name) +".csv", "a+") as f:

                if map_state.cur_pose is not None:
                    temp_cure_pose = "{} {} {} {}".format(map_state.cur_pose[0], map_state.cur_pose[1],
                                map_state.cur_pose[2], map_state.cur_pose[3])

                if len(map_state.poses) > 0:
                    poses = map_state.poses[-1]
                    temp_poses = "{} {} {} {}".format(poses[0], poses[1], poses[2], poses[3])

                for i, p in enumerate(map.get_all_detecting_point()):
                    d = np.where((color == p.color).all(axis=1))
                    list_points_c.append([d[0][0] , p.pt[0], p.pt[1], p.pt[2]])

                f.writelines("{};{};{};{}\n".format(frame_id,
                                                    temp_poses,
                                                    temp_cure_pose,
                                                    list_points_c))

        num_map_points = map.num_points()
        if num_map_points>0:
            for i,p in enumerate(map.get_all_detecting_point()):
                if p.pt[2] < c_p[2]:
                    point_after_camera.append(p.pt)
                    point_color_after_camera.append(np.flip(p.color))
                map_state.points.append(p.pt)
                map_state.colors.append(np.flip(p.color))


        map_state.points = np.array(map_state.points)
        # print( "dodano do widoku {} punktów".format(map_state.points.size/3))
        #map_state.colors = np.array(map_state.colors)/256.
        #map_state.colors = np.array(map_state.colors) * 0.
        map_state.colors = np.array(map_state.colors)

        # point clustering
        if len(point_after_camera) > 2:
            X = np.array(point_after_camera)
            Z = ward(pdist(X[:, 0::2]))
            cluster = fcluster(Z, t=9.5, criterion='distance')
            # cluster = fclusterdata(X[:, 0::2], t=6)
            unique, count = np.unique(cluster, return_counts=True)
            # if frame_id % 5 == 0:
            for n, v in zip(unique, count):
                if v > 4:
                    res = np.where(cluster == n)
                    point_after_camera = np.array(point_after_camera)
                    point_color_after_camera = np.array(point_color_after_camera)
                    t = point_after_camera[res]
                    c = point_color_after_camera[res]
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

                    # pose = np.identity(4)
                    # pose[:3, 3] = np.array([t_xmin, t_ymin, t_zmin])
                    # size = np.array([t_xmax, t_ymax, t_zmax])
                    # print(pose)
                    # print(size)



        for kf in keyframes:
            for kf_cov in kf.get_covisible_by_weight(kMinWeightForDrawingCovisibilityEdge):
                if kf_cov.kid > kf.kid:
                    map_state.covisibility_graph.append([*kf.Ow, *kf_cov.Ow])
            if kf.parent is not None: 
                map_state.spanning_tree.append([*kf.Ow, *kf.parent.Ow])
            for kf_loop in kf.get_loop_edges():
                if kf_loop.kid > kf.kid:
                    map_state.loops.append([*kf.Ow, *kf_loop.Ow])                
        map_state.covisibility_graph = np.array(map_state.covisibility_graph)   
        map_state.spanning_tree = np.array(map_state.spanning_tree)   
        map_state.loops = np.array(map_state.loops)

        self.qmap.put(map_state)


    def draw_vo(self, vo):
        if self.qvo is None:
            return
        vo_state = Viewer3DVoElement()
        vo_state.poses = np.array(vo.poses)
        vo_state.traj3d_est = np.array(vo.traj3d_est).reshape(-1,3)
        vo_state.traj3d_gt = np.array(vo.traj3d_gt).reshape(-1,3)        
        
        self.qvo.put(vo_state)


    def updateTwc(self, pose):
        self.Twc.m = pose


    @staticmethod
    def drawPlane(num_divs=200, div_size=10):
        # Plane parallel to x-z at origin with normal -y
        minx = -num_divs*div_size
        minz = -num_divs*div_size
        maxx = num_divs*div_size
        maxz = num_divs*div_size
        #gl.glLineWidth(2)
        #gl.glColor3f(0.7,0.7,1.0)
        gl.glColor3f(0.7,0.7,0.7)
        gl.glBegin(gl.GL_LINES)
        for n in range(2*num_divs):
            gl.glVertex3f(minx+div_size*n,0,minz)
            gl.glVertex3f(minx+div_size*n,0,maxz)
            gl.glVertex3f(minx,0,minz+div_size*n)
            gl.glVertex3f(maxx,0,minz+div_size*n)
        gl.glEnd()


