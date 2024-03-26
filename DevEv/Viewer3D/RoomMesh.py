import os
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from OpenGL.GLU import *
import pkg_resources
from .TexturedMesh import OBJ, GLMeshTexturedItem, MTL

TOY_MAPPING = {
    'pink_ballon':'pink_ball', 'tower_bloc':'red_tower', 'cylinder_tower':'tower', 'ball_container':'bucket',
}


def compute_bounding_box_center(vertices):
    # Convert the list of vertices to a NumPy array for easier manipulation
    vertices_array = np.array(vertices)

    # Compute the minimum and maximum coordinates along each axis (x, y, z)
    min_coords = np.min(vertices_array, axis=0)
    max_coords = np.max(vertices_array, axis=0)

    # Calculate the center point by averaging the minimum and maximum coordinates along each axis
    bounding_box_center = (min_coords + max_coords) / 2

    return bounding_box_center

class RoomManager():
    def __init__(self, viewer3D):
        self.viewer3D = viewer3D
        self.toy_to_update = []
        self.room_texture = []
        self.toy_objects = {}
        self.read_room()
        
    def read_toys(self, filename= "", as_new = False):
        if not os.path.exists(filename): 
            return
        
        self.toy_to_update = []
        data = np.load(filename, allow_pickle=True).item()
        for n, obj in self.toy_objects.items():
            name = n
            if n in TOY_MAPPING: name = TOY_MAPPING[n]
            if name in data and len(data[name]) > 0:
                obj["data"] = data[name]
                frame_list = [x for x in data[name].keys() if "p3d" in data[name][x]]
                if len(frame_list) == 0: continue
                min_f = min(frame_list)
                info = data[name][min_f]
                offset = info["p3d"] - obj["center"]
                obj["center"] = info["p3d"]
                obj["default_center"] = np.copy(info["p3d"])
                obj["item"].translate(offset[0], offset[1], offset[2])
                self.toy_to_update.append(name)
        
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
        self.viewer3D.addItem(self.mtl_data)

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
            if 'map_Kd' in  mtl and "toy_" not in name:
                #continue
                #if "Carpet3.png" in mtl["map_Kd"] or "SquareMat2.png" in mtl["map_Kd"]:
                texture = {"coords":np.array(ob["textures"]).reshape(-1, 2) , "name":ob["material"][0], "mtl":self.mtl_data.contents}
                mesh_data = gl.MeshData(vertexes=vert, faces=face)
                element = GLMeshTexturedItem(meshdata=mesh_data, textures = texture, smooth=True, drawEdges=True, glOptions=option_gl)
                element.parseMeshData()
                self.room_textured.append(element)
                self.viewer3D.addItem(element)
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
                self.viewer3D.addItem(toy)
                toy.opts['drawEdges'] = False
                center = compute_bounding_box_center(vert)
                self.toy_objects[name.replace("toy_", "")] = {"item":toy, "center":center, 
                                                                "data":{}, "default_center":center}
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
        self.viewer3D.addItem(self.room)

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
    
    def update(self, f):
        for toy_name in self.toy_to_update:
            obj = self.toy_objects[toy_name]
            if not f in obj["data"]: continue
            if not "p3d" in obj["data"][f]: offset = obj["default_center"] - obj["center"]
            else: offset = obj["data"][f]["p3d"] - obj["center"]
            obj["center"] += offset
            obj["item"].translate(*offset)
        return
    
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