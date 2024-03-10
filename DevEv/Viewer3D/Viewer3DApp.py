
import pkg_resources
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os
import copy

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4

from matplotlib import cm
from scipy.spatial import ConvexHull
from scipy import stats
import trimesh
from .TexturedMesh import OBJ, GLMeshTexturedItem, MTL
from .EdgeSphere import create_semi_sphere


SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170]]

view3Dparams_room = {
    0: {'center': QVector3D(-0.389, 0.337, 0.361), 'distance': 3.7243, 'fov': 64.658, 'elevation': 42.0, 'azimuth': 450.0, 'viewport': None},
    1: {'center': QVector3D(-0.586, 0.686, 0.592), 'distance': 2.929, 'fov': 72.90, 'elevation': 41.0, 'azimuth': 540.0, 'viewport': None},
    2: {'center': QVector3D(-0.220, 2.359, -0.131), 'distance': 4.73, 'fov': 72.90, 'elevation': 38.0, 'azimuth': 269.0, 'viewport': None},
    3: {'center': QVector3D(-0.752, -0.339, -0.823), 'distance': 4.73, 'fov': 82.20, 'elevation': 45.0, 'azimuth': 359.0, 'viewport': None},
    4: {'center': QVector3D(-0.053, -2.836, -0.294), 'distance': 3.724, 'fov': 72.906, 'elevation': 50.0, 'azimuth': 391.0, 'viewport': None},
    5: {'center': QVector3D(-1.433, 1.582, -1.1241), 'distance': 4.199, 'fov': 92.693, 'elevation': 58.0, 'azimuth': 358.0, 'viewport': None},
    6: {'center': QVector3D(-0.280, 2.403, -0.318), 'distance': 3.303, 'fov': 72.906, 'elevation': 50.0, 'azimuth': 551.0, 'viewport': None},
    7: {'center': QVector3D(0.0498, -2.123, 0.0574), 'distance': 3.303, 'fov': 72.906, 'elevation': 46.0, 'azimuth': 538.0, 'viewport': None}
}

view3Dparams_mat = {
    0: {'center': QVector3D(1.018, 0.944, 0.587), 'distance': 0.781, 'fov': 92.68, 'elevation': 90.0, 'azimuth': 449.0, 'viewport': None},
    1: {'center': QVector3D(0.765, 1.283, 0.0168), 'distance': 2.043, 'fov': 72.906, 'elevation': 57.0, 'azimuth': 499.0, 'viewport': None},
    2: {'center': QVector3D(0.852, 0.767, 0.0784), 'distance': 2.043, 'fov': 72.906, 'elevation': 41.0, 'azimuth': 539.0, 'viewport': None},
    3: {'center': QVector3D(0.976, 0.873, 0.0784), 'distance': 1.812, 'fov': 72.906, 'elevation': 90.0, 'azimuth': 270.0, 'viewport': None},
    4: view3Dparams_room[4],
    5: view3Dparams_room[5],
    6: view3Dparams_room[6],
    7: view3Dparams_room[7],
}

TOY_MAPPING = {
    'pink_ball':'pink_ballon', 'tower_bloc':'red_tower', 'cylinder_tower':'tower', 'ball_container':'bucket',
}


def plane_intersect_batch(p0, u, p_co = np.array([0,0,0]), p_no= np.array([0,0,1]), epsilon=1e-6):
    dot = np.dot(p_no , u)
    valids = abs(dot) > epsilon
    if valids.sum() < 1: return None, None
    w = p0 - p_co[valids]
    fac = -(p_no[valids]*w).sum(1) / dot[valids]
    p = p0 + np.dot(fac[:, np.newaxis], u[np.newaxis, :])

    direction = ((p - p0)*u).sum(1) >= 0
    if direction.sum() < 1: return None, None
    p = p[direction]
    ind = np.where(valids)[0]
    ind = ind[np.logical_not(direction)]
    valids[ind] = False
    return p, valids



class View3D(gl.GLViewWidget):
    position_sig = pyqtSignal(np.ndarray)
    direction_sig = pyqtSignal(np.ndarray)
    attention_sig = pyqtSignal(np.ndarray)

    def __init__(self):
        super(View3D, self).__init__()    
        self.base_color = (1.0,0.0,0.0,0.7)
        self.base_color2 = (0.2,0.0,0.0,1.0)
        self.base_color_t = (0.8,0.8,0.8,1.0)   
        ## create three grids, add each to the view   
        xgrid = gl.GLGridItem()
        xgrid.setSize(x=40, y=40)
        xgrid.setSpacing(x=1.0, y=1.0)
        self.addItem(xgrid)
        self.current_item = {"head":None , "att":None, "vec":None, "cone":None, "hand":None, "skpoint":None ,"skline":None , "frame":0}
        self.acc_item = {"head":None , "att":None, "vec":None, "cone":None, "hand":None , "frame":[]}
        self.drawn_t_item = None
        self.drawn_t_point = None
        self.drawn_h_point = None
        self.drawn_hand_point = None
        self.accumulate = {}
        self.line_type = 0
        self.color_code = False
        self.add_t_P = False
        self.project_floor = False
        self.fill = None
        self.add_Head = True
        self.add_Hand = False
        self.click_enable = False
        self.a_pressed = False
        self.corrected_frames = {}
        self.toy_objects = {}
        self.default_length = 0.5
        self.segment = None

        room_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/Room.ply')
        self.mesh = trimesh.load_mesh(room_file)
        self.read_room()
        att_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/attention.txt')
        self.attention = self.read_attention(att_file)
        #self.keypoints = self.read_keypoints("DevEv/data_3d_DevEv_S07_04_Sync.npy")
        #self.draw_skeleton()
        self.init()
        
        return
       
    def reset(self):
        """
        Initialize the widget state or reset the current state to the original state.
        """

        #'center': QVector3D(-0.47168588638305664, 0.7047028541564941, 0.896643340587616)
        #'distance': 2.304011082681173,
        #'fov': 92.69362818343406
        #'elevation': 31.0
        #'azimuth': -80.0

 
        #self.opts['center'] = QVector3D(0.0,0.0,0.0)  ## will always appear at the center of the widget
        #self.opts['distance'] = 10        ## distance of camera from center
        #self.opts['fov'] = 90                ## horizontal field of view in degrees
        #self.opts['elevation'] = 0.0          ## camera's angle of elevation in degrees
        #self.opts['azimuth'] = 0.0  
        # 
        self.opts['center'] = QVector3D(0,0,0)  ## will always appear at the center of the widget
        self.opts['distance'] = 20.0         ## distance of camera from center
        self.opts['fov'] = 40                ## horizontal field of view in degrees
        self.opts['elevation'] = 90          ## camera's angle of elevation in degrees
        self.opts['azimuth'] = 0            ## camera's azimuthal angle in degrees 
                                             ## (rotation around z-axis 0 points along x-axis)
        self.opts['viewport'] = None         ## glViewport params; None == whole widget
        self.setBackgroundColor(pg.getConfigOption('background'))
        
    def init(self):
        u =  np.array([[0.0,0.0,0.0], [0.0,0.0,1.0]])
        self.current_item["head"] = gl.GLScatterPlotItem(pos = u[0].reshape(1,3), color=(0.0,0.0,1.0,1.0), size = np.array([30.0]),  glOptions = 'translucent')
        self.current_item["att"] = gl.GLScatterPlotItem(pos = u[1].reshape(1,3), color=self.base_color, size = np.array([1.0]), glOptions = 'additive')
        self.current_item["vec"] = gl.GLLinePlotItem(pos = u, color = np.array([self.base_color, self.base_color2]), width= 8.0, antialias = True, glOptions = 'additive', mode = 'lines')
        self.current_item["cone"] = self.draw_cone(u[0], u[1])
        self.current_item["hand"] = gl.GLScatterPlotItem(glOptions = 'additive')
        #c = np.array([[0.9,0.5,0.2,1.0], [0.9,1.0,0.0,1.0]])
        c = np.array([[1.0,0.0,1.0,1.0], [0.0,1.0,0.0,1.0]])
        self.current_item["hand"].setData(pos = np.zeros((2,3)), color=c, size = np.array([12.0]))

        for _, obj in self.current_item.items():
            if type(obj) == int or obj is None: continue
            obj.hide()
            self.addItem(obj)

        self.acc_item["head"] = gl.GLScatterPlotItem(pos = u[0], color=(0.6, 0.6, 1.0, 0.4), size = np.array([15.0]), glOptions = 'additive')
        self.acc_item["att"] = gl.GLScatterPlotItem(pos = u[1], color=(0.7, 0.7, 0.7, 0.2), size = np.array([1.0]), glOptions = 'translucent')
        self.acc_item["vec"] = gl.GLLinePlotItem(pos = u, color =(0.7, 0.7, 0.7, 0.2), width= 3.0, antialias = True, glOptions = 'additive', mode = 'lines')
        #d, _ = self.draw_Ncone(u[:1], u[1:])
        self.acc_item["cone"] = []
        self.acc_item["hand"] = gl.GLScatterPlotItem(glOptions = 'translucent')
        #c = np.array([[1.0,0.5,1.0,1.0], [0.5,1.0,0.5,1.0]])
        c = np.array([[1.0,0.5,1.0,1.0], [0.0, 0.5, 0.0,1.0]])
        self.acc_item["hand"].setData(pos = np.zeros((2,3)), color=c, size = np.array([10.0]))

        #self.current_item["skpoint"].hide()
        #self.current_item["skline"].hide()
        self.acc_item["hand"].hide()
        
        for _, obj in self.acc_item.items():
            if type(obj) == list or obj is None: continue
            obj.hide()
            self.addItem(obj)
                    
        #md = gl.MeshData.sphere(rows=10, cols=20)
        self.semi_sphere = {"item":create_semi_sphere(radius = 0.4, num_segments = 9),
                            "center":np.array([0,0,0]), 
                            "show":False}
        self.semi_sphere["item"].hide()
        self.addItem(self.semi_sphere["item"])      
              
        return

    def set3DView(self, view_id, cam_type):
        params = view3Dparams_room
        if cam_type == 1:
            params = view3Dparams_mat
        self.opts.update(params[view_id])
        return

    def keyPressEvent(self, event):
        if not self.click_enable:
            event.ignore()
            return
        
        translate = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        step = 4.0
        if translate: step = 10.0
        if event.key() == Qt.Key_Left:
            self.modify_attention_click([-step, 0.0], translate)
        elif event.key() == Qt.Key_Right:
            self.modify_attention_click([step, 0.0], translate)
        elif event.key() == Qt.Key_Up:
            if not translate: step = -step
            self.modify_attention_click([0, -step], translate)
        elif event.key() == Qt.Key_Down:
            if not translate: step = -step
            self.modify_attention_click([0.0, step], translate)
        if event.key() == Qt.Key_A:
            self.a_pressed = True
        else:
            event.ignore()
            return
        event.accept()
        return

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_A:
            self.a_pressed = False
            event.accept()
        else:
            event.ignore()
            
    def mousePressEvent(self, ev):   
        self.mousePos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not (self.a_pressed and self.click_enable): 
            ev.ignore()
            return
        if ev.buttons() == Qt.MouseButton.LeftButton:
            #if (ev.modifiers() & Qt.KeyboardModifier.AltModifier):
            p = self.mousePos
            
            viewport = np.array(list(self.getViewport()), dtype=np.int32)
            # Get the modelview matrix
            modelview = np.array(self.viewMatrix().data(), dtype=np.float64)
            
            # Get the projection matrix
            projection = np.array(self.projectionMatrix().data(), dtype=np.float64)
            print(projection.shape, p.x(), p.y(), modelview.shape, viewport)
            # Unproject the pixel coordinate to obtain the ray in world space
            x = int(p.x())
            y = int(viewport[-1]) - int(p.y())
            near_point = gluUnProject(x, y, 0.0, modelview, projection, viewport)
            #near_point = gluUnProject(p.x(),  viewport[-1] - p.y(), 0.0, modelview, projection, viewport)
            far_point = gluUnProject(p.x(),  viewport[-1] - p.y(), 1.0, modelview, projection, viewport)
            near_point = np.array(near_point[:3])
            far_point = np.array(far_point[:3])
            # Calculate the ray direction
            ray_direction = (far_point - near_point)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            #ray_direction[-1] = -ray_direction[-1]
            

            camera_pos = self.cameraPosition()
            camera_pos = np.array([camera_pos.x(), camera_pos.y(), camera_pos.z()])
            
            new_att = self.collision(camera_pos, ray_direction)
            if new_att is None: return
            dx, dy, dz = new_att - self.current_item["att"].pos[0]
            self.translate_attention_p(dx, dy, dz)
            ev.accept()
        else: ev.ignore()
        return 

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        diff = lpos - self.mousePos
        self.mousePos = lpos
        
        if ev.buttons() == Qt.MouseButton.LeftButton:
            if (ev.modifiers() & Qt.KeyboardModifier.AltModifier):
                if self.click_enable: self.modify_attention_click([diff.x(), diff.y()], False)
            elif (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                if self.click_enable: self.modify_attention_click([diff.x(), diff.y()], True)
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == Qt.MouseButton.RightButton:
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        elif ev.buttons() == Qt.MouseButton.MiddleButton:
            self.pan(diff.x(), diff.y(), 0, relative='view')
      
    def modify_attention_click(self, diff, translate):  
        R = self.viewMatrix()   
        if translate:
            scale_factor = self.pixelSize(self.opts['center'] )
            translation = scale_factor*R.transposed().mapVector(QVector3D(diff[0], -diff[1], 0.0))
            self.translate_head(translation[0], translation[1], 0, emit = True)
        else:
            rotation_speed = 7.0
            sensitivity = self.pixelSize(self.opts['center'])
            right_vector = QVector3D(R[0, 0], R[0, 1], R[0, 2])
            up_vector = QVector3D(R[1, 0], R[1, 1], R[1, 2])
            forward_vector = np.array([R[2, 0], R[2, 1], R[2, 2]])

            head = self.current_item["head"].pos[0]
            u =  self.current_item["att"].pos[0] -head
        
            s = np.sign(np.dot(forward_vector, u))
            angle_x = s *diff[0] * rotation_speed * sensitivity  # Example rotation angle for x-axis
            angle_y = s * diff[1] * rotation_speed * sensitivity  # Example rotation angle for y-axis

            # Construct quaternions using camera's axis vectors
            q = QQuaternion.fromAxisAndAngle(up_vector, angle_x)  # Rotate around camera's up vector
            q *= QQuaternion.fromAxisAndAngle(right_vector, angle_y)  # Rotate around camera's right vector

            self.rotate_attention_signal(q, True)

        return

    @pyqtSlot(bool)
    def set_annotation(self, state):
        self.click_enable = state
        return

    def read_room(self):
        option_gl = {
        GL_LIGHTING:True,
        GL_LIGHT0:True,
        GL_LIGHT1:True,
        GL_DEPTH_TEST: True,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        GL_POLYGON_SMOOTH:True,
        #'glShadeModel': (GL_FLAT),

        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
        GL_COLOR_MATERIAL: True,
        'glColorMaterial' : (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE),

        'glMaterialfv':( GL_FRONT_AND_BACK, GL_AMBIENT, (1.0, 1.0, 1.0, 1.0) ),
        'glMaterialfv':( GL_FRONT_AND_BACK, GL_DIFFUSE, (0.9, 0.9, 0.9, 1.0) ),
        #'glMaterialfv':( GL_FRONT_AND_BACK, GL_SPECULAR, (0.18, 0.18, 0.18, 1.0) ),
        'glMaterialf':( GL_FRONT_AND_BACK, GL_SHININESS, 0.0),


        'glLightfv' : (GL_LIGHT0, GL_POSITION,  (-0.5, 1.8, 1.0, 1.0)), # point light from the left, top, front
        'glLightfv' : (GL_LIGHT1, GL_AMBIENT, (0.6, 0.6, 0.6, 1.0)),
        'glLightfv': (GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1.0)),
        #'glLightfv': (GL_LIGHT1, GL_SPECULAR, (0.6, 0.6, 0.6, 1)),

        'glLight' : (GL_LIGHT1, GL_POSITION,  (0.7, -1.6, 1.0, 1.0)), # point light from the left, top, front
        'glLightfv' : (GL_LIGHT1, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0)),
        'glLightfv': (GL_LIGHT1, GL_DIFFUSE, (1, 1, 1, 1)),
        #'glLightfv': (GL_LIGHT1, GL_SPECULAR, (0.5, 0.5, 0.5, 1)),
        }  

        mat_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/Room.obj')
        mtl_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/Room.mtl')
        obj = OBJ(mat_file, swapyz=True)
        self.mtl_data = MTL(filename=mtl_file)
        self.addItem(self.mtl_data)

        vertices = []
        faces = []
        #normals = []
        colors = []
        self.room_textured = []
        self.toy_objects = {}
        count = 0
        for name, ob in obj.content.items():
            if len(ob["material"]) == 0: 
                print(name, np.array(ob["vertexes"]).shape)
                continue
            if "camera" in name: continue
            
            vert = np.array(ob["vertexes"]).reshape(-1, 3)
            face = np.array(ob["faces"]).reshape(-1, 3)
            #normal = np.array(ob["normals"]).reshape(-1, 3)

            mtl = self.mtl_data.contents[ob["material"][0]]
            if 'map_Kd' in  mtl:
                #continue
                #if "Carpet3.png" in mtl["map_Kd"] or "SquareMat2.png" in mtl["map_Kd"]:
                texture = {"coords":np.array(ob["textures"]).reshape(-1, 2) , "name":ob["material"][0], "mtl":self.mtl_data.contents}
                mesh_data = gl.MeshData(vertexes=vert, faces=face)
                element = GLMeshTexturedItem(meshdata=mesh_data, textures = texture, smooth=True, drawEdges=True, glOptions=option_gl)
                element.parseMeshData()
                self.room_textured.append(element)
                self.addItem(element)
                continue
            
            color = []
            counts = ob["count"]
            counts.append(vert.shape[0])
            counts = np.array(counts)
            counts = counts[1:] - counts[:-1]
            for m, n in zip(ob["material"], counts):
                mtl= self.mtl_data.contents[m]
                c = np.array([mtl["Kd"][0], mtl["Kd"][1],mtl["Kd"][2], 1.0])
                if c[0] > 0.8 and c[1] > 0.8 and c[2] > 0.8:
                    #c = np.array(c - np.array([0.05,0.05,0.05,0.05]))
                    c = np.array(np.array([0.6,0.6,0.6,1.0]))
                #if "pCube13" in name:
                #c = np.array(np.array([0.5,0.5,0.5,0.9]))
                c = np.repeat(c[None,:], n, axis = 0)
                color.append(c)
            #print(name, ob["material"], count, face[-1][-1])
            color = np.concatenate(color, axis = 0)
            
            if "toy" in name: 
                mesh_data = gl.MeshData(vertexes=vert, faces=face, vertexColors=color)
                toy = gl.GLMeshItem(meshdata=mesh_data, smooth=True, drawEdges=True, glOptions=option_gl, edgeColor=(0.6, 0.1, 0.1, 1.0))
                toy.parseMeshData()
                self.addItem(toy)
                toy.opts['drawEdges'] = False
                self.toy_objects[name.replace("toy_", "")] = {"item":toy, "center":np.mean(vert, axis = 0), "data":{}}
                continue
            
            vertices.append(vert)
            colors.append(color)
            #normals.append(normal)
            faces.append(face + count)
            count += face[-1][-1] + 1
            
        vertices = np.concatenate(vertices, axis = 0)
        faces = np.concatenate(faces, axis = 0)
        colors = np.concatenate(colors, axis = 0)   
        #normals = np.concatenate(normals, axis = 0)  
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors)

        self.room = gl.GLMeshItem(meshdata=mesh_data, smooth=True, drawEdges=True, glOptions=option_gl)
        self.room.parseMeshData()
        self.addItem(self.room)

        for item in self.room_textured:
            item.opts['drawFaces'] = True
            item.opts['drawEdges'] = False
        self.room.opts['drawFaces'] = True
        self.room.opts['drawEdges'] = False

        """#print(obj.faces)
        vert = np.array(obj.vertexes).reshape(-1, 3)
        face = np.array(obj.face_ind).reshape(-1, 3)
        texture = np.array(obj.textures).reshape(-1, 2)

        print(vert.shape, face.shape, texture.shape, obj.mtl)
        mesh_data = gl.MeshData(vertexes=vert, faces=face)
        textures = {"coords":texture, "mtl":obj.mtl}
        #self.room = gl.GLMeshItem(meshdata=mesh_data, smooth=True, drawEdges=False, glOptions='translucent')
        self.room = GLMeshTexturedItem(meshdata=mesh_data, textures = textures, smooth=True, drawEdges=False, glOptions='translucent')
        self.addItem(self.room)"""
        return 

    def accumulate3D(self, state):
        self.accumulate = state
        u =  np.array([[0.0,0.0,0.0], [0.0,0.0,1.0]])
        self.acc_item["head"].setData(pos = u[0])
        self.acc_item["head"].hide()
        self.acc_item["vec"].setData(pos = u)
        self.acc_item["vec"].hide()
        self.acc_item["att"].setData(pos = u[1], size = np.array([1.0]))
        self.acc_item["att"].hide()
        self.acc_item["hand"].setData(pos = np.zeros((2,3)))
        self.acc_item["hand"].hide()
        for i in self.acc_item["cone"]:
            self.removeItem(i)
        self.acc_item["cone"] = []
        self.acc_item["frame"] = []
        return

    def collision(self, P, U):
        if self.semi_sphere["show"]:
            sphere = self.semi_sphere["item"].opts['meshdata']
            v = sphere.vertexes() + self.semi_sphere["center"].reshape(-1, 3)
            f = sphere.faces()

            mesh = trimesh.Trimesh(vertices=v, faces=f)
            intersection, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=P.reshape(-1, 3), ray_directions=U.reshape(-1, 3))
            if len(intersection) > 0:
                d = np.sqrt(np.sum((P - intersection) ** 2, axis=1))
                ind = np.argsort(d)
                P = self.semi_sphere["center"]
                U = intersection[ind[0]] - P
    
        if self.mesh is None: return None
        
        ray_origins = P.reshape(-1, 3)
        ray_directions = U.reshape(-1, 3)

        # Get the intersections
        intersection, index_ray, index_tri = self.mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions)

        if len(intersection) > 0:
            d = np.sqrt(np.sum((P - intersection) ** 2, axis=1))
            ind = np.argsort(d)
            return intersection[ind[0]]

        return None

    def setRoomStyle(self, view):
        if view == 0:
            self.clearRoom(True)
        else:
            self.clearRoom(False)
        
        if view == 1:
            # Wireframe
            #item.updateGLOptions(self, opts)
            for item in self.room_textured:
                item.opts['drawFaces'] = False
                item.opts['drawEdges'] = True
            self.room.opts['drawFaces'] = False
            self.room.opts['drawEdges'] = True
            self.room.updateGLOptions({GL_DEPTH_TEST: True})
        elif view == 2:
            # Transparent
            for item in self.room_textured:
                item.opts['drawFaces'] = True
                item.opts['drawEdges'] = False
                #item.setOpacity(0.2)
            self.room.opts['drawFaces'] = True
            self.room.opts['drawEdges'] = False
            c = self.room.colors
            c[:,-1] = 0.4
            self.room.setColor(c)
            self.room.updateGLOptions({GL_DEPTH_TEST: False})
        elif view == 3:
            for item in self.room_textured:
                item.opts['drawFaces'] = True
                item.opts['drawEdges'] = False

            self.room.opts['drawFaces'] = True
            self.room.opts['drawEdges'] = False
            self.room.updateGLOptions({GL_DEPTH_TEST: True})
            c = self.room.colors
            c[:,-1] = 1.0
            self.room.setColor(c)
        return

    def clearRoom(self, state):
        self.room.setVisible(not state)
        for obj in self.room_textured:
            obj.setVisible(not state)
        for n, obj in self.toy_objects.items():
            obj["item"].setVisible(not state)        
        return

    def setDome(self, state):
        self.semi_sphere["show"] = state
        self.draw_frame(None, plot_vec=True)
        return

    def draw_cone(self, p0, p1, L = 2.5, n=8, R= 0.6, just_data = False):
        # vector in direction of axis
        R0, R1 = 0, R
        v = p1 - p0
        # find magnitude of vector
        mag = np.linalg.norm(v)
        # unit vector in direction of axis
        v = v / mag
        # make some vector not in the same direction as v
        not_v = np.array([0, 1, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # print n1,'\t',norm(n1)
        # normalize n1
        n1 /= np.linalg.norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, L, n)
        theta = np.linspace(0, 2 * np.pi, n+1)
        # use meshgrid to make 2d arrays
        
        X, Y, Z = [p0[i] + v[i] * L + R *
                np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

        P = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        p = np.append([p0], P, axis =0) 
        faces1 = np.array([[0,i+1, i] for i in range(len(P))])
        faces1[-1,-2] = 1
        d = gl.MeshData(vertexes=p, faces=faces1)
        if just_data: return d
        #d.setFaceColors(self.base_color)
        cone = gl.GLMeshItem(meshdata=d, glOptions = 'translucent', drawEdges=True, computeNormals=False, color=self.base_color)   
        return cone

    def draw_Ncone(self, p0_list, p1_list, L = 2.5, n=8, R= 0.6):
        # vector in direction of axis
        R0, R1 = 0, R
        
        vertexes = []
        faces_list = []
        for k, (p0, p1) in enumerate(zip(p0_list, p1_list)):
            v = p1 - p0
            # find magnitude of vector
            mag = np.linalg.norm(v)
            # unit vector in direction of axis
            v = v / mag
            # make some vector not in the same direction as v
            not_v = np.array([0, 1, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            # make vector perpendicular to v
            n1 = np.cross(v, not_v)
            # print n1,'\t',norm(n1)
            # normalize n1
            n1 /= np.linalg.norm(n1)
            # make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            # surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, L, n)
            theta = np.linspace(0, 2 * np.pi, n+1)
            # use meshgrid to make 2d arrays
            
            X, Y, Z = [p0[i] + v[i] * L + R *
                    np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

            P = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
            faces1 = np.array([[0,i+1, i] for i in range(len(P))])
            faces1[-1,-2] = 1
            p = np.append([p0], P, axis =0) 

            offset = len(p)
            vertexes.append(p)
            faces_list.append(faces1 + k*offset)

        vertexes = np.concatenate(vertexes)
        faces_list = np.concatenate(faces_list)
        d = gl.MeshData(vertexes=vertexes, faces=faces_list)

        return d, offset

    def draw_skeleton(self):
        self.current_item["skpoint"] = gl.GLScatterPlotItem(glOptions = 'additive')
        c = np.array(CocoColors)/255.0
        r = np.copy(c[:,2])
        b = np.copy(c[:,0])
        c[:,0] = r
        c[:,2] = b

        self.current_item["skpoint"].setData(pos = np.zeros((17,3)), color=c, size = np.array([15.0]))
        self.current_item["skpoint"].setVisible(True)
        self.addItem(self.current_item["skpoint"])

        self.current_item["skline"] = gl.GLLinePlotItem(pos =  np.zeros((34,3)), color = (1.0,0.0,0.0,1.0), width= 3.0, glOptions = 'additive', mode = 'lines')
        self.current_item["skline"].setVisible(True)
        self.addItem(self.current_item["skline"])


        return

    def read_toys(self, filename= "", as_new = False):
        if not os.path.exists(filename): 
            return
        
        data = np.load(filename, allow_pickle=True).item()
        for n, obj in self.toy_objects.items():
            name2 = n
            if n in TOY_MAPPING: name2 = TOY_MAPPING[n]
            if name2 in data and len(data[name2]) > 0:
                obj["data"] = data[name2]
                min_f = min(data[name2].keys())
                info = data[name2][min_f]
                offset = info["p3d"] - obj["center"]
                obj["center"] = info["p3d"]
                obj["item"].translate(offset[0], offset[1], offset[2])
        
        return
    
    def read_attention(self, filename= "DevEv/metadata/RoomData/attention.txt", as_new=False):
        if not os.path.exists(filename): 
            return
        attention = {}
        xyz = []
        self.corrected_frames = {}
        self.corrected_frames_hand = {}
        with open(filename, "r") as f:
            data = f.readlines()

        segment = []
        start, old_frame = None, None
        for i, d in enumerate(data):
            d_split = d.replace("\n", "").split(",")
            xhl, yhl, zhl, xhr, yhr, zhr = 0,0,0,0,0,0
            flag, flag_h = 0, 0
            if len(d_split)== 10:
                frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d_split
            elif len(d_split)== 11:
                frame, b0, b1, b2, A0, A1, A2, att0, att1, att2, flag = d_split
            elif len(d_split)== 18:
                frame, flag, flag_h, b0, b1, b2, A0, A1, A2, att0, att1, att2, xhl, yhl, zhl, xhr, yhr, zhr = d_split
            elif len(d_split) < 10: continue
            else:
                print("Error in attention file")
                exit()
            flag, flag_h = int(flag), int(flag_h)
            pos = np.array([float(att0), float(att1), float(att2)])
            #vec = np.array([float(A0), float(A1), float(A2)])
            b = np.array([float(b0), float(b1), float(b2)])
            handL = np.array([float(xhl), float(yhl), float(zhl)])
            handR = np.array([float(xhr), float(yhr), float(zhr)])
            color_time = cm.jet(i / len(data))

            att_line = np.array([b, pos])
            size = np.linalg.norm(pos - b)
            if as_new:
                flag, flag_h = 0, 0
            if flag > 0: self.corrected_frames[int(frame)]= flag
            if flag_h > 0: self.corrected_frames_hand[int(frame)]= flag_h
            if size < 1e-6: 
                attention[int(frame)] = np.copy(attention[int(frame) - 1]).item()
                xyz.append(xyz[-1])
                continue
            vec = (pos - b)/ ( size + 1e-6)
            att_vec = np.array([b, b + self.default_length*vec]) 
            size = np.clip(size*2.0, 5.0, 60.0)
             
            attention[int(frame)] = {"u":att_vec, "line":att_line, "head":b, "att":pos, "corrected_flag":flag, "c_density":(1.0,1.0,1.0,1.0),
                                    "c_time":color_time, "size":size, "corrected_flag_hand":flag_h, "handL":handL,"handR":handR}
        
            xyz.append(pos)
            # Get segments
            if start is None: start = int(frame)
            if old_frame is None: 
                old_frame = int(frame)
                continue
            if abs(int(frame)-old_frame) > 50:
                segment.append((start,old_frame))
                start = int(frame)
            old_frame = int(frame)    
        segment.append((start, int(frame)))  
        self.segment = segment     
        print("Segments", segment)    
            
        print("Attention Loaded with", len([x for x, y in self.corrected_frames.items() if y == 1]), "already corrected frames")
        print("Attention Loaded with", len([x for x, y in self.corrected_frames.items() if y == 2]), "frames selected for correction")
        print("Hands Loaded with", len([x for x, y in self.corrected_frames_hand.items() if y == 1]), "already corrected frames")
        print("Hands Loaded with", len([x for x, y in self.corrected_frames_hand.items() if y == 2]), "frames selected for correction")
        print(len(attention), "frames in file")
        
        """xyz = np.array(xyz, dtype = float)
        kde = stats.gaussian_kde(xyz.T)
        density = kde(xyz.T)   
        a, b = min(density), max(density)
        density = (density - a) / (b-a + 1e-6)
        density = cm.jet(density)
        
        for i, (f, info) in enumerate(attention.items()):
            info["c_density"] = density[int(i)]   """          
        return attention

    def read_keypoints(self, filename):
        if not os.path.exists(filename): return {}
        output = {}
        data = np.load(filename, allow_pickle=True).item()
        # print(data.keys()[0])
        for f, data1 in data.items():
            p = data1['3d']
            if len(p) == 0: continue
            bones = []
            for p1, p2 in SKELETON:
                #bones.append([p[p1], p[p2]])
                bones.append(p[p1])
                bones.append(p[p2])
            bones = np.array(bones)
            hand = data1['3d_closest_points'][9:11]
            output[f] = {"p":np.array(p), "l":bones, "hand":hand}
        return output
    
    def draw_frame(self, f, plot_vec = False):
        if f is None:
            f = self.current_item["frame"]                              

            #self.current_item["skpoint"].setData(pos = self.keypoints[f]["p"])
            #self.current_item["skline"].setData(pos = self.keypoints[f]["l"])
            #self.current_item["skline"].setVisible(True)
            #self.current_item["skpoint"].setVisible(True)
        for n, obj in self.toy_objects.items():
            if not "data" in obj: continue
            if not f in obj["data"]: continue
            info = obj["data"][f]
            offset = info["p3d"] - obj["center"]
            obj["center"] = obj["center"] + offset
            obj["item"].translate(offset[0], offset[1], offset[2])
                
                
        if self.attention is None or f not in self.attention: 
            return
                
        att = self.attention[f]["att"]
        size_p = self.attention[f]["size"].reshape(1)
        self.current_item["att"].setData(pos = att.reshape(1,3), size = size_p)
        self.current_item["att"].setVisible(plot_vec and not self.line_type == 3)

        head = self.attention[f]["head"]
        self.current_item["head"].setData(pos = head.reshape(1,3))
        self.current_item["head"].setVisible(self.add_Head)
        
        # Hands
        self.current_item["hand"].setData(pos = [self.attention[f]["handL"], self.attention[f]["handR"]])
        self.current_item["hand"].setVisible(self.add_Hand)

        # Semi-Sphere
        offset = head - self.semi_sphere["center"]
        self.semi_sphere["item"].translate(offset[0], offset[1], offset[2])
        self.semi_sphere["center"] = head
        self.semi_sphere["item"].setVisible(self.semi_sphere["show"])
        
        u = self.attention[f]["u"]
        if self.line_type == 1:
            self.current_item["vec"].setData(pos = self.attention[f]["line"])
        else:
            self.current_item["vec"].setData(pos = u)
        self.current_item["vec"].setVisible(plot_vec and self.line_type in [0,1])

        cone_data = self.draw_cone(u[0], u[1], just_data=True)
        self.current_item["cone"].setMeshData(meshdata=cone_data)
        self.current_item["cone"].setVisible(plot_vec and self.line_type == 2)
        self.current_item["frame"] = f

        if self.accumulate and f not in self.acc_item["frame"]:
            if len(self.acc_item["frame"]) == 0:
                acc_heads = np.copy(head).reshape(1,3)
                acc_vecs = np.copy(self.current_item["vec"].pos).reshape(2,3)
                acc_att = np.copy(att).reshape(1,3)
                acc_size = np.copy(size_p).reshape(1)
                acc_hands = np.copy(self.current_item["hand"].pos).reshape(2,3)

                #if f in self.keypoints and self.keypoints[f]["hand"] is not None: 
                #    acc_hand = np.copy(self.keypoints[f]["hand"]).reshape(2,3)
                #    self.acc_item["hand"].setVisible(False)
                self.acc_item["head"].setVisible(self.add_Head)
                self.acc_item["vec"].setVisible(plot_vec and self.line_type in [0,1])
                self.acc_item["att"].setVisible(plot_vec and not self.line_type == 3)      
                self.acc_item["hand"].setVisible(self.add_Hand) 
            else:
                acc_heads = np.concatenate([np.copy(self.acc_item["head"].pos), np.copy(head).reshape(1,3)])
                acc_vecs = np.concatenate([np.copy(self.acc_item["vec"].pos), np.copy(self.current_item["vec"].pos).reshape(2,3)])
                acc_att = np.concatenate([np.copy(self.acc_item["att"].pos), np.copy(att).reshape(1,3)])
                acc_size = np.concatenate([np.copy(self.acc_item["att"].size) , np.copy(size_p).reshape(1)])
                acc_hands = np.concatenate([np.copy(self.acc_item["hand"].pos), np.copy(self.current_item["hand"].pos).reshape(2,3)])
                #if f in self.keypoints and self.keypoints[f]["hand"] is not None: acc_hand = np.concatenate([np.copy(self.acc_item["hand"].pos) , np.copy(self.keypoints[f]["hand"]).reshape(2,3)])

            
            self.acc_item["head"].setData(pos = acc_heads)
            self.acc_item["vec"].setData(pos = acc_vecs)
            self.acc_item["att"].setData(pos = acc_att, size = acc_size)
            #color_hands = np.array([[1.0,0.5,1.0,0.2], [0.5,1.0,0.5,0.2]])
            color_hands = np.array([[1.0,0.0,1.0,1.0], [0.0, 0.5, 0.0,1.0]])
            self.acc_item["hand"].setData(pos = acc_hands, color = np.tile(color_hands, (acc_hands.shape[0]//2, 1)))
            
            #if f in self.keypoints and self.keypoints[f]["hand"] is not None: self.acc_item["hand"].setData(pos = acc_hand)
            self.acc_item["frame"].append(f)
  
            if plot_vec and self.line_type == 2:
                #c = (0.0, 0.7, 0.0, 0.5)
                c = (0.7, 0.7, 0.7, 0.2)
                item = gl.GLMeshItem(meshdata=cone_data, glOptions = 'translucent', drawEdges=False, computeNormals=False, color=c)
                #item.setTransform(self.current_item["cone"].transform())
                self.acc_item["cone"].append(item)
                self.addItem(item)
        return

    def showAll(self, frame_min, frame_max, as_type):
        self.clear_t()

        total = len([0 for f in range(frame_min, frame_max) if f in self.attention])
        points = []
        vecs = []
        color, color_hands = [], []
        count = 0
        heads = []
        hands = []
        size_list = []
        for f in range(frame_min, frame_max):
            if f not in self.attention: 
                continue
            if self.color_code == 1:
                color.append(list( cm.jet(count/ (1.0*total)) ))
            else:
                color.append(list(self.base_color_t))
            points.append(self.attention[f]["att"])
            heads.append(self.attention[f]["head"])
            if self.line_type == 0:
                u = self.attention[f]["u"]
            else: #elif self.line_type == 1:
                u = self.attention[f]["line"]
            size_list.append(self.attention[f]["size"])
            vecs.append(u[0])
            vecs.append(u[1])
            hands.append(self.attention[f]["handL"])
            hands.append(self.attention[f]["handR"])
            #color_hands.extend([[1.0, 0.5, 1.0, 0.5],[0.5,1.0,0.5,0.5]])
            color_hands.extend([[1.0,0.0,1.0,1.0], [0.0, 0.5, 0.0,1.0]])
            count += 1

        if total == 0: 
            return

        heads = np.array(heads)
        hands = np.array(hands)
        points = np.array(points)
        vecs = np.array(vecs)
        color = np.array(color)
        size_list = np.array(size_list)
        color_head = (0.6,0.6, 1.0, 0.7)
        if len(points) > 3 and self.color_code == 2:
            kde = stats.gaussian_kde(points.T)
            density = kde(points.T)   
            a, b = min(density), max(density)
            density = (density - a) / (b-a + 1e-6)
            color = cm.jet(density)
        if self.color_code in [1,2]: color_head = color

        itemh = gl.GLScatterPlotItem(pos = heads, color=color_head, size = 13.0, glOptions = 'additive')
        item = gl.GLScatterPlotItem(pos = points, color=color, size = size_list, glOptions = 'translucent')
        itemhands = gl.GLScatterPlotItem(pos = hands, color=np.array(color_hands), size = 10.0, glOptions = 'translucent')

        self.drawn_t_point = item 
        self.drawn_h_point = itemh      
        self.drawn_hand_point = itemhands    
        if self.add_t_P: self.addItem(item) 
        if self.add_Head: 
            self.addItem(itemh) 
            head_L = np.linalg.norm(heads[1:] - heads[:-1], axis = -1)
            print("Head Trajectory Length: {:.3f} m".format(head_L.sum()))
        if self.add_Hand: self.addItem(itemhands)

        if as_type == 0:    # Vector type
            if self.project_floor: vecs[:,2] = 0.0
            
            if self.line_type == 2: # cone type
                color[:, -1] = 0.2
                d, n = self.draw_Ncone(vecs[::2], vecs[1::2])
                color = np.repeat(color, n, axis=0)
                d.setVertexColors(color)
                item = gl.GLMeshItem(meshdata=d, glOptions = 'translucent', drawEdges=True, antialias=True, computeNormals=False)   
            else:    
                color = np.repeat(color, 2, axis=0)
                color[:,-1] = 0.5
                item = gl.GLLinePlotItem(pos = vecs, color = color, width= self.default_length, antialias=True, glOptions='translucent', mode='lines')
            
            if self.line_type != 3: 
                self.drawn_t_item = item
                self.addItem(item) 

        else: # Hull type
            if total < 3: return 
            color[:, -1] = 0.3
            color = np.repeat(color, 2, axis=0)
            
            hull = ConvexHull(vecs)
            info_str = "Hull Volume: " + str(hull.volume) + " m3"
            if self.project_floor:
                vecs[:,2] = 0.0 
                hull2d = ConvexHull(vecs[:,:2])
                info_str = "Hull Surface Area: " + str(hull2d.area/2.0) + " m2"
            print(info_str)
            np.save("mesh_hull.npy", {"vertexes":vecs, "faces":hull.simplices})
                
            if hull.good is not None:
                d = gl.MeshData(vertexes=vecs[hull.good], faces=hull.simplices[hull.good])
                color = color[hull.good]
            else:
                d = gl.MeshData(vertexes=vecs, faces=hull.simplices)
            
            d.setVertexColors(color)
            mesh = gl.GLMeshItem(meshdata=d, glOptions='translucent')               
            if self.line_type != 3: 
                self.drawn_t_item = mesh
                self.addItem(mesh) 
                
        return

    def clear_t(self):
        if self.drawn_t_item is not None:
            self.removeItem(self.drawn_t_item)
        if self.drawn_t_point is not None and self.add_t_P:
            self.removeItem(self.drawn_t_point)
        if self.drawn_h_point is not None and self.add_Head:
            self.removeItem(self.drawn_h_point)
        if self.drawn_hand_point is not None and self.add_Hand:
            self.removeItem(self.drawn_hand_point)

        self.drawn_t_point, self.drawn_t_item, self.drawn_h_point, self.drawn_hand_point = None, None, None, None

        return

    def clear_fill(self):
        if self.fill is not None:
            self.removeItem(self.fill)
        self.fill = None
        return

    def colorCheck(self, state):
        self.color_code = state
        for f, item in self.current_item.items(): 
            if type(item) == int or item is None: continue
            color = self.base_color
            f = self.current_item["frame"]
            if state == 1:
                color = self.attention[f]["c_time"]
            elif state == 2:
                color = self.attention[f]["c_density"]
            if isinstance(item, gl.GLMeshItem):
                item.setColor(color)
            else:
                item.setData(color = color)
        return

    def addPCheck(self, state):
        
        if state:
            if self.drawn_t_point is not None:
                self.addItem(self.drawn_t_point) 
        else:
            if self.drawn_t_point is not None and self.add_t_P:
                self.removeItem(self.drawn_t_point)  
        self.add_t_P = state           
        return

    def addHeadCheck(self, state):
        if state:
            if self.drawn_h_point is not None:
                self.addItem(self.drawn_h_point) 
            
        else:
            if self.drawn_h_point is not None and self.add_Head:
                self.removeItem(self.drawn_h_point)  
        self.add_Head = state 
        self.current_item["head"].setVisible(state)      
        return

    def addHandCheck(self, state):
        if state:
            if self.drawn_hand_point is not None:
                self.addItem(self.drawn_hand_point) 
            pass
        else:
            if self.drawn_hand_point is not None and self.add_Hand:
                self.removeItem(self.drawn_hand_point)  
            pass
        self.add_Hand = state 
        self.current_item["hand"].setVisible(state)      
        return

    def fill_acc(self, state):
        
        if not state:
            self.clear_fill()
        else:
            if not self.accumulate: return
            drawn_f = sorted(list(self.acc_item["frame"]))
            new_lines = []

            for i in range(len(drawn_f) - 1):
                f1 = drawn_f[i]
                f2 = drawn_f[i+1]
                if self.line_type == 0:
                    u1 = self.attention[f1]["u"]
                    u2 = self.attention[f2]["u"]
                else:
                    u1 = self.attention[f1]["line"]
                    u2 = self.attention[f2]["line"]
                p0 = (u1[0] + u2[0])/2.0
                p1 = u1[1]
                p2 = u2[1]
                d = np.linalg.norm(p1-p2)
                if d < 0.2: continue
                for j in range(20):
                    t = j/40.0
                    p = p1 * t + p2*(1-t)
                    vec = (p - p0)/ np.linalg.norm(p - p0)
                    att_vec = np.array([p0, p0 + self.default_length*vec]) 
                    new_lines.append(att_vec)

            """for i in np.arange(0.2, 0.8, 0.1):
                for j in np.arange(-1.0, 1.0, 0.1):
                    for k in np.arange(-1.0, 0.7, 0.1):
                        p = np.array([i +0.01*np.random.randn(),j+0.01*np.random.randn(),k+0.01*np.random.randn()])
                        vec = p/ np.linalg.norm(p)
                        att_vec = np.array([p0, p0 + self.default_length*vec]) 
                        new_lines.append(att_vec)"""


            new_lines = np.array(new_lines)
            line = gl.GLLinePlotItem(pos = new_lines, color = (0.8, 0.8, 0.8, 0.15), width= 3.0, antialias=True)
            
            if self.line_type != 3:  
                self.fill = line
                self.addItem(line)
        return

    def modify_attention(self, frame):
        if not frame in self.attention:
            return False
        data = self.attention[frame]
        data["head"] = np.copy(self.current_item["head"].pos[0])
        u = self.current_item["att"].pos - data["head"]
        u = u / np.linalg.norm(u)
    
        att = self.collision(data["head"], u)
        if att is None: return False
        data["line"][0] = data["head"]
        data["line"][1] = att
        data["att"] = att
        size = np.linalg.norm(att - data["head"])

        data["size"] = np.clip(size*4.0, 10.0, 80.0)
        data["u"][0] = np.copy(data["head"])
        data["u"][1] = np.copy(data["head"] + self.default_length*u)

        return att

    def save_hands(self, frame):
        if not frame in self.attention:
            return False
        data = self.attention[frame]
        handL_changed, handR_changed = False, False
        if np.linalg.norm(data["handL"] - self.current_item["hand"].pos[0]) > 1e-4:
            handL_changed = True
        if np.linalg.norm(data["handR"] - self.current_item["hand"].pos[1]) > 1e-4:
            handR_changed = True
        data["handL"] = np.copy(self.current_item["hand"].pos[0])
        data["handR"] = np.copy(self.current_item["hand"].pos[1])
        return handL_changed, handR_changed

    def translate_head(self, dx, dy, dz, emit=False):
        new_pos = self.current_item["head"].pos[0] + np.array([dx, dy, dz])
        self.current_item["head"].setData(pos=new_pos.reshape(1,3))

        u = np.copy(self.current_item["att"].pos[0] - new_pos)
        u = u / np.linalg.norm(u)
    
        if self.line_type == 0:
            self.current_item["vec"].setData(pos=[new_pos, new_pos + self.default_length*u])
        else:
            self.current_item["vec"].setData(pos=[new_pos, self.current_item["att"].pos[0]])

        # Semi Sphere
        self.semi_sphere["item"].translate(dx, dy, dz)
        self.semi_sphere["center"] = new_pos

        cone_data = self.draw_cone(new_pos, new_pos+u, just_data=True)
        self.current_item["cone"].setMeshData(meshdata=cone_data)
        
        if emit:
            self.position_sig.emit(new_pos)
            self.direction_sig.emit(u)
            return

        return u

    def translate_attention_p(self, dx, dy, dz):
        head = self.current_item["head"].pos[0]
        new_pos = np.copy(self.current_item["att"].pos[0] + np.array([dx, dy, dz]))
        self.current_item["att"].setData(pos=new_pos.reshape(1,3))

        u = new_pos - head
        u = u / np.linalg.norm(u)

        if self.line_type == 0:
            self.current_item["vec"].setData(pos=[head, head + self.default_length*u])
        else:
            self.current_item["vec"].setData(pos=[head, self.current_item["att"].pos[0]])

        cone_data = self.draw_cone(head, head+u, just_data=True)
        self.current_item["cone"].setMeshData(meshdata=cone_data)
        return u 

    def rotate_attention(self, angle, axis, modify_att):

        head = self.current_item["head"].pos[0]
        u =  self.current_item["att"].pos[0] - head
        u = rotate(u, angle*np.pi/180.0, axis)
        if modify_att:
            self.current_item["att"].setData(pos= np.copy(head + u).reshape(1,3))

        if self.line_type in [0,3]:
            u = self.default_length * u / np.linalg.norm(u)
            self.current_item["vec"].setData(pos= [head, head + u])
        elif self.line_type == 1:
            self.current_item["vec"].setData(pos= [head, head + self.current_item["att"].pos[0]])


        cone_data = self.draw_cone(head, head+u, just_data=True)
        self.current_item["cone"].setMeshData(meshdata=cone_data)

        return self.current_item["att"].pos[0]

    def rotate_attention_signal(self, M, modify_att):
        head = self.current_item["head"].pos[0]
        u =  self.current_item["att"].pos[0] -head
        u = M.rotatedVector(QVector3D(u[0], u[1], u[2]))
        u = np.array([u[0], u[1], u[2]])
        
        if modify_att:
            self.current_item["att"].setData(pos= (head + u).reshape(1,3))

        if self.line_type in [0,3]:
            u = self.default_length * u / np.linalg.norm(u)
            self.current_item["vec"].setData(pos= [head, head + u])
        elif self.line_type == 1:
            self.current_item["vec"].setData(pos=[head, self.current_item["att"].pos[0]])
        
        cone_data = self.draw_cone(head, head+u, just_data=True)
        self.current_item["cone"].setMeshData(meshdata=cone_data)


        self.attention_sig.emit(self.current_item["att"].pos[0])
        self.direction_sig.emit(u)
        return 

    def translate_hand_left(self, dx, dy, dz, emit=False):
        #old_handl = self.current_item["handL"].pos[0]
        new_pos = self.current_item["hand"].pos[0] + np.array([dx, dy, dz])
        self.current_item["hand"].setData(pos=[new_pos, self.current_item["hand"].pos[1]])

        if emit:
            self.position_sig.emit(new_pos)
            return
        return

    def translate_hand_right(self, dx, dy, dz, emit=False):
        #old_handl = self.current_item["handL"].pos[0]
        new_pos = self.current_item["hand"].pos[1] + np.array([dx, dy, dz])
        self.current_item["hand"].setData(pos=[self.current_item["hand"].pos[0], new_pos])

        if emit:
            self.position_sig.emit(new_pos)
            return
        return
    
def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': return np.dot(X, np.array([
    [1.,  0,  0],
    [0 ,  c, -s],
    [0 ,  s,  c]
  ]))
  elif axis == 'y': return np.dot(X, np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
  ]))
  elif axis == 'z': return np.dot(X, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
  ]))




