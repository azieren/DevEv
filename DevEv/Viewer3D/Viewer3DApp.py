
import pkg_resources
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os
import cv2

from OpenGL.GL import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QVector3D, QQuaternion
from plyfile import PlyData, PlyElement

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
        self.base_color_t = (0.8,0.8,0.8,1.0)   
        ## create three grids, add each to the view   
        xgrid = gl.GLGridItem()
        xgrid.setSize(x=50, y=40)
        self.addItem(xgrid)
        self.drawn_item = {}
        self.drawn_t_item = None
        self.drawn_t_point = None
        self.drawn_h_point = None
        self.accumulate = {}
        self.line_type = 0
        self.color_code = False
        self.add_t_P = False
        self.project_floor = False
        self.old_f = None
        self.fill = None
        self.add_Head = True
        self.click_enable = False

        #plane_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/room_setup2.json')
        #self.plane_dict = self.read_planes(plane_file)
        room_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/LAB.ply')
        self.mesh = self.read_room(room_file)
        att_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/attention.txt')
        self.attention = self.read_attention(att_file)
        #self.keypoints = self.read_keypoints("data_3d_DevEv_S_12_01_MobileInfants_trim.npy")
        #self.draw_skeleton()
        return

    
    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        old_pos = self.mousePos
        diff = lpos - self.mousePos
        self.mousePos = lpos
        
        if ev.buttons() == Qt.MouseButton.LeftButton:
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == Qt.MouseButton.MiddleButton:
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        elif ev.buttons() == Qt.MouseButton.RightButton and self.click_enable:
            self.modify_attention_click([diff.x(), diff.y()], ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            

    def modify_attention_click(self, diff, translate):  
        R = self.viewMatrix()   
        if translate:
            scale_factor = self.pixelSize(self.opts['center'] )
            translation = scale_factor*R.transposed().mapVector(QVector3D(diff[0], -diff[1], 0.0))
            self.translate_attention(self.old_f, translation[0], translation[1], 0, emit = True)
        else:
            q = QQuaternion.fromAxisAndAngle(R[2,0], R[2,1], R[2,2], -diff[0])
            q *= QQuaternion.fromAxisAndAngle(R[0,0], R[0,1], R[0,2], -diff[1])
            self.rotate_attention_signal(self.old_f, q, True)
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
        'glMaterialf':( GL_FRONT_AND_BACK, GL_SHININESS, 1.45),

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

        mat_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/LAB.obj')
        mtl_file = pkg_resources.resource_filename('DevEv', 'metadata/RoomData/scene/LAB.mtl')
        obj = OBJ(mat_file, swapyz=True)
        self.mtl_data = MTL(filename=mtl_file)
        self.addItem(self.mtl_data)

        vertices = []
        faces = []
        colors = []
        self.room_textured = []
        count = 0
        for name, ob in obj.content.items():
            if "material" not in ob: continue
                #print(name, np.array(ob["vertexes"]).shape)
                #exit()
            mtl = self.mtl_data.contents[ob["material"]]
            vert = np.array(ob["vertexes"]).reshape(-1, 3)
            face = np.array(ob["faces"]).reshape(-1, 3)
            color = np.array([mtl["Kd"][0], mtl["Kd"][1],mtl["Kd"][2], 1.0])
            
            if color[0] > 0.8 and color[1] > 0.8 and color[2] > 0.8:
                color = np.array([0.65, 0.65, 0.65, 1.0])

            if 'map_Kd' in  mtl:
                #if "Carpet3.png" in mtl["map_Kd"] or "SquareMat2.png" in mtl["map_Kd"]:
                texture = {"coords":np.array(ob["textures"]).reshape(-1, 2) , "name":ob["material"], "mtl":self.mtl_data.contents}
                mesh_data = gl.MeshData(vertexes=vert, faces=face)
                element = GLMeshTexturedItem(meshdata=mesh_data, textures = texture, smooth=True, drawEdges=False, glOptions=option_gl)
                self.room_textured.append(element)
                self.addItem(element)
                continue
            #print(name, ob["material"], count, face[-1][-1])
            color = np.repeat(color[None,:], vert.shape[0], axis = 0)
            vertices.append(vert)
            colors.append(color)
            faces.append(face + count)
            count += face[-1][-1] + 1
            
        vertices = np.concatenate(vertices, axis = 0)
        faces = np.concatenate(faces, axis = 0)
        colors = np.concatenate(colors, axis = 0)     
        print(vertices.shape, colors.shape)  
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors)

        self.room = gl.GLMeshItem(meshdata=mesh_data, smooth=True, drawEdges=False, glOptions=option_gl)
        self.addItem(self.room)
        

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
        self.clear()
        self.accumulate = state
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
        cone = gl.GLMeshItem(meshdata=d, glOptions = 'translucent', drawEdges=True, antialias=True, computeNormals=False, color=self.base_color)   
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
 
        self.sk_point = gl.GLScatterPlotItem()
        c = np.array(CocoColors)/255.0
        r = np.copy(c[:,2])
        b = np.copy(c[:,0])
        c[:,0] = r
        c[:,2] = b

        self.sk_point.setData(pos = self.keypoints[f]["p"], color=c, size = 10.0)
        self.addItem(self.sk_point)

        self.sk_lines = gl.GLLinePlotItem(pos = self.keypoints[f]["l"], color = (1.0,0.0,0.0, 1.0), width= 2.0, antialias=True)
        self.addItem(self.sk_lines)
        return

    def read_attention(self, filename= "DevEv/metadata/RoomData/attention.txt"):
        if not os.path.exists(filename): return
        attention = {}
        with open(filename, "r") as f:
            data = f.readlines()

        for i, d in enumerate(data):
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d.replace("\n", "").split(",")
            pos = np.array([float(att0), float(att1), float(att2)])
            #vec = np.array([float(A0), float(A1), float(A2)])
            b = np.array([float(b0), float(b1), float(b2)])
            color_time = cm.jet(i / len(data))

            att_line = np.array([b, pos])
            size = np.linalg.norm(pos - b)
            if size < 1e-6: 
                attention[int(frame)] = attention[int(frame) - 1]
                continue
            vec = (pos - b)/ ( size + 1e-6)
            att_vec = np.array([b, b + 5.0*vec]) 
            attention[int(frame)] = {"u":att_vec, "line":att_line, "head":b, "att":pos,
                                    "c_time":color_time, "size": size*4.0}

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
        output = {}
        data = np.load(filename, allow_pickle=True).item()
        for f, p in data.items():
            bones = []
            for p1, p2 in SKELETON:
                bones.append([p[p1], p[p2]])
            output[f] = {"p":np.array(p), "l":np.array(bones)}
        return output

    def draw_frame(self, f, plot_vec = False):
        if not self.accumulate:
            self.clear()
        else:
            if self.old_f in self.drawn_item:
                c = (0.7, 0.7, 0.7, 0.35)
                for n, item in self.drawn_item[self.old_f].items(): 
                    if isinstance(item, gl.GLMeshItem):
                        item.setColor(c)
                    else:
                        item.setData(color = c)
                    item.setGLOptions('translucent')
                        
                        

        if f not in self.attention: 
            return
        
        """if f in self.keypoints:
            self.sk_point.setData(pos = self.keypoints[f]["p"])
            self.sk_lines.setData(pos = self.keypoints[f]["l"])"""


        att = self.attention[f]["att"]
        size_p = self.attention[f]["size"]
        item_att = gl.GLScatterPlotItem(pos = att, color=self.base_color, size = np.array([size_p]))
        self.addItem(item_att)
        
        head = self.attention[f]["head"]
        item_head = gl.GLScatterPlotItem(pos = head, color=self.base_color, size = np.array([20]), glOptions = 'translucent')

        if not self.add_Head:
            item_head.hide()
        self.addItem(item_head)

        if self.line_type in [0,3]:
            u = self.attention[f]["u"]
            line = gl.GLLinePlotItem(pos = u, color = self.base_color, width= 2.0, antialias=True, glOptions = 'translucent')
        elif self.line_type == 1:
            u = self.attention[f]["line"]
            line = gl.GLLinePlotItem(pos = u, color = self.base_color, width= 2.0, antialias=True, glOptions = 'translucent')
        else:
            [u1, u2] = self.attention[f]["u"]
            line = self.draw_cone(u1, u2) 

        if plot_vec and self.line_type == 3: 
            line.hide()
            item_att.hide()

        self.addItem(line) 
        

        self.drawn_item[f] = {"head":item_head, "att":item_att, "vec":line}
        self.old_f = f
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
            if self.line_type == 2: 
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

    def clear(self):
        for f, item_list in self.drawn_item.items(): 
            for _, item in item_list.items(): self.removeItem(item)
        self.drawn_item = {}
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
        if state == 1:
            for f, item_list in self.drawn_item.items(): 
                for _, item in item_list.items():
                    item.setData(color=self.attention[f]["c_time"])
        elif state == 2:
            for f, item_list in self.drawn_item.items(): 
                for _, item in item_list.items():
                    item.setData(color=self.attention[f]["c_density"])            
        else:
            for f, item_list in self.drawn_item.items(): 
                for i_, item in item_list.items():
                    item.setData(color=self.base_color)

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
        if self.old_f in self.drawn_item:
            self.drawn_item[self.old_f]["head"].setVisible(state)        
        return

    def fill_acc(self, state):
        
        if not state:
            self.clear_fill()
        else:
            if not self.accumulate: return
            drawn_f = sorted(list(self.drawn_item.keys()))
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
            line = gl.GLLinePlotItem(pos = new_lines, color = (0.8, 0.8, 0.8, 0.15), width= 2.0, antialias=True)
            
            if self.line_type != 3:  
                self.fill = line
                self.addItem(line)
        return

    def modify_attention(self, frame):
        if not frame in self.attention:
            return False
        drawn_data = self.drawn_item[frame]
        data = self.attention[frame]
        data["head"] = drawn_data["head"].pos
        u = drawn_data["att"].pos - drawn_data["head"].pos
        u = u / np.linalg.norm(u)
    
        att = self.collision(drawn_data["head"].pos, u)
        data["line"][0] = drawn_data["head"].pos
        data["line"][1] = att
        data["att"] = att
        u = att - data["head"]
        u = u / np.linalg.norm(u)
        data["u"][0] = drawn_data["head"].pos
        data["u"][1] = drawn_data["head"].pos + 5.0*u

        return att

    def translate_attention(self, frame, dx, dy, dz, emit=False):
        if not frame in self.drawn_item:
            return False
        data = self.drawn_item[frame]
        new_pos = data["head"].pos + np.array([dx, dy, dz])
        data["head"].setData(pos=new_pos)

        u = data["att"].pos - new_pos
        u = u / np.linalg.norm(u)

        if self.line_type in [0,3]:
            data["vec"].setData(pos=[new_pos, new_pos + 5.0*u])
        elif self.line_type == 1:
            data["vec"].setData(pos=[new_pos, data["att"].pos])

        if emit:
            self.position_sig.emit(new_pos)
            self.direction_sig.emit(u)
            return

        return u

    def translate_attention_p(self, frame, dx, dy, dz):
        if not frame in self.drawn_item:
            return False
        data = self.drawn_item[frame]
        d = data["att"]
        new_pos = d.pos + np.array([dx, dy, dz])
        d.setData(pos=new_pos)

        u = new_pos - data["head"].pos
        u = u / np.linalg.norm(u)

        if self.line_type in [0,3]:
            data["vec"].setData(pos=[data["head"].pos, data["head"].pos + 5.0*u])
        elif self.line_type == 1:
            data["vec"].setData(pos=[data["head"].pos, new_pos])
        return u 

    def rotate_attention(self, frame, angle, axis, modify_att):
        if not frame in self.drawn_item:
            return False
        data = self.drawn_item[frame]

        u =  data["att"].pos - data["head"].pos
        u = rotate(u, angle*np.pi/180.0, axis)
        if modify_att:
            data["att"].setData(pos= data["head"].pos + u)
        if self.line_type in [0,3]:
            u = 5.0 * u / np.linalg.norm(u)
            data["vec"].setData(pos=[data["head"].pos, data["head"].pos + u])
        elif self.line_type == 1:
            data["vec"].setData(pos=[data["head"].pos, data["att"].pos])
        return data["att"].pos

    def rotate_attention_signal(self, frame, M, modify_att):
        if not frame in self.drawn_item:
            return False
        data = self.drawn_item[frame]

        u =  data["att"].pos - data["head"].pos
        u = M.rotatedVector(QVector3D(u[0], u[1], u[2]))
        u = np.array([u[0], u[1], u[2]])
        if modify_att:
            data["att"].setData(pos= data["head"].pos + u)
        if self.line_type in [0,3]:
            u = 5.0 * u / np.linalg.norm(u)
            data["vec"].setData(pos=[data["head"].pos, data["head"].pos + u])
        elif self.line_type == 1:
            data["vec"].setData(pos=[data["head"].pos, data["att"].pos])
        
        self.attention_sig.emit(data["att"].pos)
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




