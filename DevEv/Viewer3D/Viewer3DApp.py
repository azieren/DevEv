
import pkg_resources
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os
import cv2

from OpenGL.GL import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QVector3D, QQuaternion

from matplotlib import cm
from scipy.spatial import ConvexHull
from scipy import stats
import trimesh
from .objects import get_balloon, get_red_ball, get_shovel, get_farm, get_xyl, get_tree, get_ring, get_piggy, get_red_toy
from .TexturedMesh import OBJ, GLMeshTexturedItem, MTL

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170]]

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
        self.base_color = (1.0,0.0,0.0,1.0)
        self.base_color2 = (0.2,0.0,0.0,1.0)
        self.base_color_t = (0.8,0.8,0.8,1.0)   
        ## create three grids, add each to the view   
        xgrid = gl.GLGridItem()
        xgrid.setSize(x=50, y=40)
        self.addItem(xgrid)
        self.current_item = {"head":None , "att":None, "vec":None, "cone":None , "frame":0}
        self.acc_item = {"head":None , "att":None, "vec":None, "cone":None , "frame":[]}
        self.drawn_t_item = None
        self.drawn_t_point = None
        self.drawn_h_point = None
        self.accumulate = {}
        self.line_type = 0
        self.color_code = False
        self.add_t_P = False
        self.project_floor = False
        self.fill = None
        self.add_Head = True
        self.click_enable = False
        self.corrected_frames = set()

        #plane_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/room_setup2.json')
        #self.plane_dict = self.read_planes(plane_file)
        room_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/Room.ply')
        self.mesh = self.read_room(room_file)
        att_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/attention.txt')
        self.attention = self.read_attention(att_file)
        self.keypoints = self.read_keypoints("DevEv/data_2.6d_DevEv_S07_04.npy")
        #self.draw_skeleton()
        self.init()
        return

    def init(self):
        u =  np.array([[0.0,0.0,0.0], [0.0,0.0,1.0]])
        self.current_item["head"] = gl.GLScatterPlotItem(pos = u[0].reshape(1,3), color=self.base_color, size = np.array([20.0]),  glOptions = 'additive')
        self.current_item["att"] = gl.GLScatterPlotItem(pos = u[1].reshape(1,3), color=self.base_color, size = np.array([1.0]), glOptions = 'additive')
        self.current_item["vec"] = gl.GLLinePlotItem(pos = u, color = np.array([self.base_color, self.base_color2]), width= 3.0, antialias = True, glOptions = 'additive', mode = 'lines')
        self.current_item["cone"] = self.draw_cone(u[0], u[1])

        for _, obj in self.current_item.items():
            if type(obj) == int: continue
            obj.hide()
            self.addItem(obj)

        c = (0.7, 0.7, 0.7, 0.35)
        self.acc_item["head"] = gl.GLScatterPlotItem(pos = u[0], color=c, size = np.array([20.0]), glOptions = 'additive')
        self.acc_item["att"] = gl.GLScatterPlotItem(pos = u[1], color=c, size = np.array([1.0]), glOptions = 'additive')
        self.acc_item["vec"] = gl.GLLinePlotItem(pos = u, color =c, width= 3.0, antialias = True, glOptions = 'additive', mode = 'lines')
        self.acc_item["cone"] = [] #self.draw_cone(u[0], u[1])
        #self.acc_item["cone"].setColor(c)

        for _, obj in self.acc_item.items():
            if type(obj) == list: continue
            obj.hide()
            self.addItem(obj)
        return

    def keyPressEvent(self, event):
        if not self.click_enable:
            event.accept()
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
        event.accept()
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
            q = QQuaternion.fromAxisAndAngle(R[2,0], R[2,1], R[2,2], -diff[0]*0.3)
            q *= QQuaternion.fromAxisAndAngle(R[0,0], R[0,1], R[0,2], -diff[1]*0.3)
            self.rotate_attention_signal(q, True)
        return

    @pyqtSlot(bool)
    def set_annotation(self, state):
        self.click_enable = state
        return


    def read_room(self, file):
        option_gl = {
        GL_LIGHTING:True,
        GL_LIGHT0:True,
        GL_LIGHT1:True,
        GL_DEPTH_TEST: True,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        GL_POLYGON_SMOOTH:True,



        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
        GL_COLOR_MATERIAL: True,
        'glColorMaterial' : (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE),

        'glMaterialfv':( GL_FRONT_AND_BACK, GL_AMBIENT, (1.0, 1.0, 1.0, 1.0) ),
        'glMaterialfv':( GL_FRONT_AND_BACK, GL_DIFFUSE, (0.9, 0.9, 0.9, 1.0) ),
        'glMaterialfv':( GL_FRONT_AND_BACK, GL_SPECULAR, (0.17, 0.17, 0.17, 1) ),
        'glMaterialf':( GL_FRONT_AND_BACK, GL_SHININESS, 50.0),

        'glLight' : (GL_LIGHT0, GL_POSITION,  (-7, -7, 12, 1.0)), # point light from the left, top, front
        'glLightfv' : (GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0)),
        'glLightfv': (GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1)),
        'glLightfv': (GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1)),

        'glLight' : (GL_LIGHT1, GL_POSITION,  (7, 10, 12, 1.0)), # point light from the left, top, front
        'glLightfv' : (GL_LIGHT1, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0)),
        'glLightfv': (GL_LIGHT1, GL_DIFFUSE, (1, 1, 1, 1.0)),
        'glLightfv': (GL_LIGHT1, GL_SPECULAR, (0.5, 0.5, 0.5, 1)),
        
            }  

        
        mesh = trimesh.load_mesh(file)
        mat_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/Room.obj')
        mtl_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/Room.mtl')
        obj = OBJ(mat_file, swapyz=True)
        self.mtl_data = MTL(filename=mtl_file)
        self.addItem(self.mtl_data)

        vertices = []
        faces = []
        colors = []
        self.room_textured = []
        count = 0
        for name, ob in obj.content.items():
            if len(ob["material"]) == 0: 
                print(name, np.array(ob["vertexes"]).shape)
                continue
                
                #exit()
            #print(name)

            vert = np.array(ob["vertexes"]).reshape(-1, 3)
            face = np.array(ob["faces"]).reshape(-1, 3)

            mtl = self.mtl_data.contents[ob["material"][0]]

            if 'map_Kd' in  mtl:
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
                    c = np.array([0.65, 0.65, 0.65, 1.0])
                c = np.repeat(c[None,:], n, axis = 0)
                color.append(c)
            #print(name, ob["material"], count, face[-1][-1])
            color = np.concatenate(color, axis = 0)
            vertices.append(vert)
            colors.append(color)
            faces.append(face + count)
            count += face[-1][-1] + 1
            
        vertices = np.concatenate(vertices, axis = 0)
        faces = np.concatenate(faces, axis = 0)
        colors = np.concatenate(colors, axis = 0)     
        print(vertices.shape, colors.shape)  
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
        return mesh

    def accumulate3D(self, state):
        self.accumulate = state
        u =  np.array([[0.0,0.0,0.0], [0.0,0.0,1.0]])
        self.acc_item["head"].setData(pos = u[0])
        self.acc_item["head"].hide()
        self.acc_item["vec"].setData(pos = u)
        self.acc_item["vec"].hide()
        self.acc_item["att"].setData(pos = u[1], size = np.array([1.0]))
        self.acc_item["att"].hide()
        self.acc_item["frame"] = []
        return

    def collision(self, P, U):

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
            #item.updateGLOptions(self, opts)
            for item in self.room_textured:
                item.opts['drawFaces'] = False
                item.opts['drawEdges'] = True
            self.room.opts['drawFaces'] = False
            self.room.opts['drawEdges'] = True
        elif view == 2:
            for item in self.room_textured:
                item.opts['drawFaces'] = True
                item.opts['drawEdges'] = False
                #item.setOpacity(0.2)
            self.room.opts['drawFaces'] = True
            self.room.opts['drawEdges'] = False
            c = self.room.colors
            c[:,-1] = 0.4
            self.room.setColor(c)
        elif view == 3:
            for item in self.room_textured:
                item.opts['drawFaces'] = True
                item.opts['drawEdges'] = False

            self.room.opts['drawFaces'] = True
            self.room.opts['drawEdges'] = False
            c = self.room.colors
            c[:,-1] = 1.0
            self.room.setColor(c)
        return

    def clearRoom(self, state):
        self.room.setVisible(not state)
        for obj in self.room_textured:
            obj.setVisible(not state)
        return

    def draw_cone(self, p0, p1, L = 8.0, n=8, R= 1.5):
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
        faces1 = np.array([[0,i+1, i] for i in range(len(P))])
        faces1[-1,-2] = 1
        p = np.append([p0], P, axis =0) 
        d = gl.MeshData(vertexes=p, faces=faces1)
        #d.setFaceColors(self.base_color)
        cone = gl.GLMeshItem(meshdata=d, glOptions = 'additive', drawEdges=True, computeNormals=False, color=self.base_color)   
        return cone

    def draw_Ncone(self, p0_list, p1_list, L = 8.0, n=8, R= 1.5):
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
        f = list(self.keypoints.keys())[0]
 
        self.sk_point = gl.GLScatterPlotItem(glOptions = 'additive')
        c = np.array(CocoColors)/255.0
        r = np.copy(c[:,2])
        b = np.copy(c[:,0])
        c[:,0] = r
        c[:,2] = b

        self.sk_point.setData(pos = self.keypoints[f]["p"], color=c, size = np.array([15.0]))
        self.addItem(self.sk_point)

        print(self.keypoints[f]["l"].shape)
        self.sk_lines = gl.GLLinePlotItem(pos = self.keypoints[f]["l"], color = (1.0,0.0,0.0,1.0), width= 5.0, glOptions = 'additive', mode = 'lines')
        self.addItem(self.sk_lines)
        return

    def read_attention(self, filename= "DevEv/metadata/RoomData/attention.txt"):
        if not os.path.exists(filename): return
        attention = {}
        with open(filename, "r") as f:
            data = f.readlines()

        for i, d in enumerate(data):
            d_split = d.replace("\n", "").split(",")
            if len(d_split)== 10:
                frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d_split
                flag = 0
            elif len(d_split)== 11:
                frame, b0, b1, b2, A0, A1, A2, att0, att1, att2, flag = d_split
            else:
                print("Error in attention file")
                exit()
            flag = int(flag)
            pos = np.array([float(att0), float(att1), float(att2)])
            #vec = np.array([float(A0), float(A1), float(A2)])
            b = np.array([float(b0), float(b1), float(b2)])
            color_time = cm.jet(i / len(data))

            att_line = np.array([b, pos])
            size = np.linalg.norm(pos - b)
            if size < 1e-6: 
                attention[int(frame)] = np.copy(attention[int(frame) - 1])
                continue
            vec = (pos - b)/ ( size + 1e-6)
            att_vec = np.array([b, b + 5.0*vec]) 
            size = np.clip(size*4.0, 10.0, 80.0)
            attention[int(frame)] = {"u":att_vec, "line":att_line, "head":b, "att":pos,
                                    "c_time":color_time, "size":size, "corrected_flag":flag}
            if flag: self.corrected_frames.add(int(frame))
        print("Attention Loaded with", len(self.corrected_frames), "already corrected frames")
        xyz = np.array([p["att"] for _, p in attention.items()])
        kde = stats.gaussian_kde(xyz.T)
        density = kde(xyz.T)   
        a, b = min(density), max(density)
        density = (density - a) / (b-a + 1e-6)
        density = cm.jet(density)
        for i, (_, p) in enumerate(attention.items()):
            p["c_density"] = density[i]             
        return attention

    def read_keypoints(self, filename):
        if not os.path.exists(filename): return {}
        output = {}
        data = np.load(filename, allow_pickle=True).item()
        for f, p in data.items():
            bones = []
            for p1, p2 in SKELETON:
                #bones.append([p[p1], p[p2]])
                bones.append(p[p1])
                bones.append(p[p2])
            bones = np.array(bones)
            output[f] = {"p":np.array(p), "l":bones}
        return output

    def draw_frame(self, f, plot_vec = False):
        if f is None:
            f = self.current_item["frame"]                              

        """if f in self.keypoints:
            self.sk_point.setData(pos = self.keypoints[f]["p"])
            self.sk_lines.setData(pos = self.keypoints[f]["l"])"""

        if f not in self.attention: 
            return
                
        att = self.attention[f]["att"]
        size_p = self.attention[f]["size"].reshape(1)
        old_u = self.current_item["att"].pos[0] - self.current_item["head"].pos[0]
        self.current_item["att"].setData(pos = att.reshape(1,3), size = size_p)
        self.current_item["att"].setVisible(plot_vec and not self.line_type == 3)

        old_pos = np.copy(self.current_item["head"].pos[0])
        head = self.attention[f]["head"]
        self.current_item["head"].setData(pos = head.reshape(1,3))
        self.current_item["head"].setVisible(self.add_Head)

        
        u = self.attention[f]["u"]
        if self.line_type == 1:
            self.current_item["vec"].setData(pos = self.attention[f]["line"])
        else:
            self.current_item["vec"].setData(pos = u)
        self.current_item["vec"].setVisible(plot_vec and self.line_type in [0,1])

        
        old_u = old_u/ np.linalg.norm(old_u)
        u = u[1] - u[0]
        u = u/ np.linalg.norm(u)
        v = np.cross(old_u, u)
        vn = np.linalg.norm(v)
        if vn > 1e-6:
            v = v / vn
            a = max(-1.0, min(1.0, np.dot(old_u, u)))
            a = np.arccos(a)*180.0/np.pi
            self.current_item["cone"].translate(-old_pos[0], -old_pos[1], -old_pos[2])
            self.current_item["cone"].rotate(a, v[0], v[1], v[2])
            self.current_item["cone"].translate(head[0], head[1], head[2])
        self.current_item["cone"].setVisible(plot_vec and self.line_type == 2)
        self.current_item["frame"] = f

        if self.accumulate and f not in self.acc_item["frame"]:
            if len(self.acc_item["frame"]) == 0:
                acc_heads = np.copy(head).reshape(1,3)
                acc_vecs = np.copy(self.current_item["vec"].pos).reshape(2,3)
                acc_att = np.copy(att).reshape(1,3)
                acc_size = np.copy(size_p).reshape(1)
                self.acc_item["head"].setVisible(True)
                self.acc_item["vec"].setVisible(True)
                self.acc_item["att"].setVisible(True)
            else:
                acc_heads = np.concatenate([np.copy(self.acc_item["head"].pos), np.copy(head).reshape(1,3)])
                acc_vecs = np.concatenate([np.copy(self.acc_item["vec"].pos), np.copy(self.current_item["vec"].pos).reshape(2,3)])
                acc_att = np.concatenate([np.copy(self.acc_item["att"].pos), np.copy(att).reshape(1,3)])
                acc_size = np.concatenate([np.copy(self.acc_item["att"].size) , np.copy(size_p).reshape(1)])

            
            self.acc_item["head"].setData(pos = acc_heads)
            self.acc_item["vec"].setData(pos = acc_vecs)
            self.acc_item["att"].setData(pos = acc_att, size = acc_size)
            self.acc_item["frame"].append(f)
            #print(len(self.acc_item["frame"]), acc_heads.shape, acc_vecs.shape)
        return

    def showAll(self, frame_min, frame_max, as_type):
        self.clear_t()

        total = len([0 for f in range(frame_min, frame_max) if f in self.attention])
        points = []
        vecs = []
        color = []
        count = 0
        heads = []
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
            count += 1

        if total == 0: 
            return

        heads = np.array(heads)
        points = np.array(points)
        vecs = np.array(vecs)
        color = np.array(color)
        size_list = np.array(size_list)

        if len(points) > 3 and self.color_code == 2:
            kde = stats.gaussian_kde(points.T)
            density = kde(points.T)   
            a, b = min(density), max(density)
            density = (density - a) / (b-a + 1e-6)
            color = cm.jet(density)

        itemh = gl.GLScatterPlotItem(pos = heads, color=color, size = 10.0)
        item = gl.GLScatterPlotItem(pos = points, color=color, size = size_list)

        self.drawn_t_point = item 
        self.drawn_h_point = itemh       
        if self.add_t_P: self.addItem(item) 
        if self.add_Head: self.addItem(itemh) 
        if as_type == 0:    # Vector type
            if self.project_floor: vecs[:,2] = 0.0
            color[:, -1] = 0.2
            if self.line_type == 2: # cone type
                d, n = self.draw_Ncone(vecs[::2], vecs[1::2])
                color = np.repeat(color, n, axis=0)
                d.setVertexColors(color)
                item = gl.GLMeshItem(meshdata=d, glOptions = 'translucent', drawEdges=True, antialias=True, computeNormals=False)   
            else:    
                color = np.repeat(color, 2, axis=0)
                item = gl.GLLinePlotItem(pos = vecs, color = color, width= 5.0, antialias=True, glOptions='translucent', mode='lines')
            
            if self.line_type != 3: 
                self.drawn_t_item = item
                self.addItem(item) 

        else: # Hull type
            if total < 3: return 
            color[:, -1] = 0.3
            color = np.repeat(color, 2, axis=0)
            hull = ConvexHull(vecs)
            if self.project_floor: vecs[:,2] = 0.0
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


            np.save("mesh.npy", {"vertexes":vecs, "faces":hull.simplices})
            print("Hull Surface Area: ", hull.area)
            print("Hull Volume: ", hull.volume)

                
        return

    def clear_t(self):
        if self.drawn_t_item is not None:
            self.removeItem(self.drawn_t_item)
        if self.drawn_t_point is not None and self.add_t_P:
            self.removeItem(self.drawn_t_point)
        if self.drawn_h_point is not None and self.add_Head:
            self.removeItem(self.drawn_h_point)

        self.drawn_t_point, self.drawn_t_item, self.drawn_h_point = None, None, None

        return

    def clear_fill(self):
        if self.fill is not None:
            self.removeItem(self.fill)
        self.fill = None
        return

    def colorCheck(self, state):
        self.color_code = state
        for f, item in self.current_item.items(): 
            if type(item) == int: continue
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
                    att_vec = np.array([p0, p0 + 5.0*vec]) 
                    new_lines.append(att_vec)

            """for i in np.arange(0.2, 0.8, 0.1):
                for j in np.arange(-1.0, 1.0, 0.1):
                    for k in np.arange(-1.0, 0.7, 0.1):
                        p = np.array([i +0.01*np.random.randn(),j+0.01*np.random.randn(),k+0.01*np.random.randn()])
                        vec = p/ np.linalg.norm(p)
                        att_vec = np.array([p0, p0 + 5.0*vec]) 
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
        data["line"][0] = data["head"]
        data["line"][1] = att
        data["att"] = att
        u = att - data["head"]
        size = np.linalg.norm(u)
        u = u / size
        data["size"] = np.clip(size*4.0, 10.0, 80.0)
        data["u"][0] = np.copy(data["head"])
        data["u"][1] = np.copy(data["head"] + 5.0*u)

        return att

    def translate_head(self, dx, dy, dz, emit=False):
        old_head = self.current_item["head"].pos[0]
        new_pos = self.current_item["head"].pos[0] + np.array([dx, dy, dz])
        old_u = self.current_item["att"].pos[0] - old_head
        self.current_item["head"].setData(pos=new_pos.reshape(1,3))

        u = np.copy(self.current_item["att"].pos[0] - new_pos)
        u = u / np.linalg.norm(u)
    
        if self.line_type == 0:
            self.current_item["vec"].setData(pos=[new_pos, new_pos + 5.0*u])
        else:
            self.current_item["vec"].setData(pos=[new_pos, self.current_item["att"].pos[0]])

        old_u = old_u/ np.linalg.norm(old_u)
        v = np.cross(old_u, u)
        vn = np.linalg.norm(v)
        if vn > 1e-5:
            v = v / vn
            a = max(-1.0, min(1.0, np.dot(old_u, u)))
            a = np.arccos(a)*180.0/np.pi
            self.current_item["cone"].translate(-old_head[0], -old_head[1], -old_head[2])
            self.current_item["cone"].rotate(a, v[0], v[1], v[2], local=False)
            self.current_item["cone"].translate(new_pos[0], new_pos[1], new_pos[2])

        if emit:
            self.position_sig.emit(new_pos)
            self.direction_sig.emit(u)
            return

        return u

    def translate_attention_p(self, dx, dy, dz):
        old_att = np.copy(self.current_item["att"].pos[0])
        head = self.current_item["head"].pos[0]
        new_pos = np.copy(self.current_item["att"].pos[0] + np.array([dx, dy, dz]))
        self.current_item["att"].setData(pos=new_pos.reshape(1,3))

        u = new_pos - head
        u = u / np.linalg.norm(u)

        
        if self.line_type == 0:
            self.current_item["vec"].setData(pos=[head, head + 5.0*u])
        else:
            self.current_item["vec"].setData(pos=[head, self.current_item["att"].pos[0]])

        old_u = old_att - head
        old_u = old_u/ np.linalg.norm(old_u)
        v = np.cross(old_u, u)
        vn = np.linalg.norm(v)
        if vn > 1e-5:
            v = v / vn
            a = max(-1.0, min(1.0, np.dot(old_u, u)))
            a = np.arccos(a)*180.0/np.pi
            self.current_item["cone"].translate(-head[0], -head[1], -head[2])
            self.current_item["cone"].rotate(a, v[0], v[1], v[2], local=False)
            self.current_item["cone"].translate(head[0], head[1], head[2])
        return u 

    def rotate_attention(self, angle, axis, modify_att):

        head = self.current_item["head"].pos[0]
        u =  self.current_item["att"].pos[0] - head
        old_u = np.copy(u)
        u = rotate(u, angle*np.pi/180.0, axis)
        if modify_att:
            self.current_item["att"].setData(pos= np.copy(head + u).reshape(1,3))

        if self.line_type in [0,3]:
            u = 5.0 * u / np.linalg.norm(u)
            self.current_item["vec"].setData(pos= [head, head + u])
        elif self.line_type == 1:
            self.current_item["vec"].setData(pos= [head, head + self.current_item["att"].pos[0]])

        old_u = old_u/ np.linalg.norm(old_u)
        u = u/np.linalg.norm(u)
        v = np.cross(old_u, u)
        vn = np.linalg.norm(v)
        if vn > 1e-5:
            v = v / vn
            a = max(-1.0, min(1.0, np.dot(old_u, u)))
            a = np.arccos(a)*180.0/np.pi
            self.current_item["cone"].translate(-head[0], -head[1], -head[2])
            self.current_item["cone"].rotate(a, v[0], v[1], v[2], local=False)
            self.current_item["cone"].translate(head[0], head[1], head[2])

        return self.current_item["att"].pos[0]

    def rotate_attention_signal(self, M, modify_att):
        head = self.current_item["head"].pos[0]
        u =  self.current_item["att"].pos[0] -head
        old_u = np.copy(u)
        u = M.rotatedVector(QVector3D(u[0], u[1], u[2]))
        u = np.array([u[0], u[1], u[2]])
        
        if modify_att:
            self.current_item["att"].setData(pos= (head + u).reshape(1,3))

        if self.line_type in [0,3]:
            u = 5.0 * u / np.linalg.norm(u)
            self.current_item["vec"].setData(pos= [head, head + u])
        elif self.line_type == 1:
            self.current_item["vec"].setData(pos=[head, self.current_item["att"].pos[0]])
        

        old_u = old_u/ np.linalg.norm(old_u)
        u = u/np.linalg.norm(u)
        v = np.cross(old_u, u)
        vn = np.linalg.norm(v)
        
        if vn > 1e-5:
            v = v / vn      
            a = max(-1.0, min(1.0, np.dot(old_u, u)))  
            a = np.arccos(a)*180.0/np.pi
            self.current_item["cone"].translate(-head[0], -head[1], -head[2])
            self.current_item["cone"].rotate(a, v[0], v[1], v[2], local=False)
            self.current_item["cone"].translate(head[0], head[1], head[2])


        self.attention_sig.emit(self.current_item["att"].pos[0])
        self.direction_sig.emit(u)
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




