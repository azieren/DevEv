import pkg_resources
from OpenGL.GL import *  # noqa
from OpenGL.GLU import *
import numpy as np
import cv2 
from PyQt5 import QtGui
import os
from pyqtgraph import makeRGBA
from pyqtgraph.opengl import GLGraphicsItem, MeshData, shaders
#import imageio

__all__ = ['GLMeshItem']

class MTL(GLGraphicsItem.GLGraphicsItem):
    def __init__(self, filename):
        GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.filename = filename

        dirname = os.path.dirname(self.filename)
        self.contents = {}
        mtl = None

        for line in open(self.filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'map_d': continue
            if values[0] == 'newmtl':
                mtl = self.contents[values[1]] = {}
            elif mtl is None:
                raise ValueError
            elif values[0] == 'map_Kd':
                # extract the texture file path and offset values from the line
                path = values[-1]
                offset = [0.0, 0.0, 0.0]
                scale = [1.0, 1.0, 1.0]
                for i in range(1, len(values) - 1):
                    if values[i] == '-s':
                        offset = [float(values[i+1]), float(values[i+2]), float(values[i+3])]
                    elif values[i] == '-o':
                        scale = [float(values[i+1]), float(values[i+2]), float(values[i+3])]


                # load the texture referred to by this declaration
                mtl[values[0]] = {
                    'path': pkg_resources.resource_filename('DevEv', os.path.join('metadata/RoomData/scene/', path.replace("\\","/"))),
                    'offset': offset,
                    'scale': scale
                }

            else:
                mtl[values[0]] = [float(x) for x  in values[1:]]

    def initializeGL(self):  
        ids = {}
        
        #glEnable(GL_TEXTURE_2D)
        for k, v in self.contents.items():
            if not type(v) == dict: continue
            if not 'map_Kd' in v: continue
            path = v['map_Kd']["path"]
            if path in ids: 
                v['texture_Kd'] = ids[path]
                continue
            image = cv2.cvtColor( cv2.imread(path),  cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 0)
            #image = imageio.imread(path)
            image = makeRGBA(image)[0]
            image[:,:,3] = 255
            shape = image.shape
            texid = v['texture_Kd'] = glGenTextures(1)
            ids[path] = texid
            print(path, shape, texid)

            glBindTexture(GL_TEXTURE_2D, texid)
            data = np.ascontiguousarray(image)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,  shape[1], shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glBindTexture(GL_TEXTURE_2D, 0)
            # Set the texture coordinate generation coefficients using glTexGenfv
            glTexGenfv(GL_S, GL_OBJECT_PLANE, [1.0, 0.0, 0.0, v['map_Kd']['offset'][0]])
            glTexGenfv(GL_T, GL_OBJECT_PLANE, [0.0, 1.0, 0.0, v['map_Kd']['offset'][1]])


        
    def initializeGLOld(self):  
        ids = {}    
        glEnable(GL_TEXTURE_2D)
        for k, v in self.contents.items():
            if not type(v) == dict: continue
            if not 'map_Kd' in v: continue
            path = v['map_Kd']
            if path in ids: 
                v['texture_Kd'] = ids[path]
                continue
            image = cv2.cvtColor( cv2.imread(path),  cv2.COLOR_BGR2RGB)
            #image = imageio.imread(path)
            image = makeRGBA(image)[0]
            image[:,:,3] = 255
            shape = image.shape
            texid = v['texture_Kd'] = glGenTextures(1)
            ids[path] = texid
            print(path, shape, texid)
            glBindTexture(GL_TEXTURE_2D, texid)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)



            #glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            """if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])"""
            
            data = np.ascontiguousarray(image.transpose((1,0,2)))
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        #glDisable(GL_TEXTURE_2D)

        return 

    def paint(self): 
        #glDisable(GL_TEXTURE_2D)
        return

class OBJ:
    def __init__(self, filename, swapyz=False):
        dirname = os.path.dirname(filename)
        """Loads a Wavefront OBJ file. """
        vertices = []
        normals = []
        textures = []
        count = 0
        self.content = {}

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'o':
                curr = values[1]
                self.content[curr] = {"vertexes":[], "textures":[], "faces":[], "normals":[], "material":[], "count":[]}
                count = 0
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], -v[2], v[1]
                vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], -v[2], v[1]
                normals.append(v)
            elif values[0] == 'vt':
                vt = [float(x) for x in values[1:3]]
                #vt = 1-vt[1], vt[0]
                textures.append(vt)
            elif values[0] in ('usemtl', 'usemat'):
                self.content[curr]["material"].append(values[1])
                self.content[curr]["count"].append(len(self.content[curr]["vertexes"]))
            elif values[0] == 'mtllib':
                self.mtl = os.path.join(dirname ,values[1])
            elif values[0] == 'f':
                face = []
                texcoords_ = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords_.append(int(w[1]))
                    else:
                        texcoords_.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)                
                self.content[curr]["vertexes"].extend([vertices[x-1]  for x in face])
                self.content[curr]["textures"].extend([textures[x-1] for x in texcoords_])                
                self.content[curr]["normals"].extend([normals[x-1]  for x in norms])
                self.content[curr]["faces"].extend([count, count+1, count +2])
                count += 3
        return
       
class GLMeshTexturedItem(GLGraphicsItem.GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays a 3D triangle mesh. 
    """
    def __init__(self, **kwds):
        """
        ============== =====================================================
        **Arguments:**
        meshdata       MeshData object from which to determine geometry for 
                       this item.
        color          Default face color used if no vertex or face colors 
                       are specified.
        edgeColor      Default edge color to use if no edge colors are
                       specified in the mesh data.
        drawEdges      If True, a wireframe mesh will be drawn. 
                       (default=False)
        drawFaces      If True, mesh faces are drawn. (default=True)
        shader         Name of shader program to use when drawing faces.
                       (None for no shader)
        smooth         If True, normal vectors are computed for each vertex
                       and interpolated within each face.
        computeNormals If False, then computation of normal vectors is 
                       disabled. This can provide a performance boost for 
                       meshes that do not make use of normals.
        ============== =====================================================
        """
        self.opts = {
            'meshdata': None,
            'color': (1., 1., 1., 1.),
            'drawEdges': False,
            'drawFaces': True,
            'edgeColor': (0.5, 0.5, 0.5, 1.0),
            'shader': None,
            'smooth': True,
            'computeNormals': True,
            'textures':None,
        }
        self.textures = kwds.pop('textures', None)
        GLGraphicsItem.GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'opaque')
        self.setGLOptions(glopts)
        shader = kwds.pop('shader', None)
        self.setShader(shader)
        
        self.setMeshData(**kwds)
        

        ## storage for data compiled from MeshData object
        self.vertexes = None
        self.normals = None
        self.colors = None
        self.faces = None 

    def setShader(self, shader):
        """Set the shader used when rendering faces in the mesh. (see the GL shaders example)"""
        self.opts['shader'] = shader
        self.update()

        
    def shader(self):
        shader = self.opts['shader']
        if isinstance(shader, shaders.ShaderProgram):
            return shader
        else:
            return shaders.getShaderProgram(shader)

    def initializeGL(self):
        if self.textures is None:
            return
        glEnable(GL_TEXTURE_2D)



    def setColor(self, c):
        """Set the default color to use when no vertex or face colors are specified."""
        self.opts['color'] = c
        self.update()

        
    def setMeshData(self, **kwds):
        """
        Set mesh data for this item. This can be invoked two ways:
        
        1. Specify *meshdata* argument with a new MeshData object
        2. Specify keyword arguments to be passed to MeshData(..) to create a new instance.
        """
        md = kwds.get('meshdata', None)
        if md is None:
            opts = {}
            for k in ['vertexes', 'faces', 'edges', 'vertexColors', 'faceColors']:
                try:
                    opts[k] = kwds.pop(k)
                except KeyError:
                    pass
            md = MeshData(**opts)
        
        self.opts['meshdata'] = md
        self.opts.update(kwds)
        self.meshDataChanged()
        self.update()
        
    
    def meshDataChanged(self):
        """
        This method must be called to inform the item that the MeshData object
        has been altered.
        """
        
        self.vertexes = None
        self.faces = None
        self.normals = None
        self.colors = None
        self.edges = None
        self.edgeColors = None
        self.update()

    
    def parseMeshData(self):
        ## interpret vertex / normal data before drawing
        ## This can:
        ##   - automatically generate normals if they were not specified
        ##   - pull vertexes/noormals/faces from MeshData if that was specified
        
        if self.vertexes is not None and self.normals is not None:
            return
        #if self.opts['normals'] is None:
            #if self.opts['meshdata'] is None:
                #self.opts['meshdata'] = MeshData(vertexes=self.opts['vertexes'], faces=self.opts['faces'])
        if self.opts['meshdata'] is not None:
            md = self.opts['meshdata']
            if self.opts['smooth'] and not md.hasFaceIndexedData():
                self.vertexes = md.vertexes()
                if self.opts['computeNormals']:
                    self.normals = md.vertexNormals()
                self.faces = md.faces()
                if md.hasVertexColor():
                    self.colors = md.vertexColors()
                if md.hasFaceColor():
                    self.colors = md.faceColors()
            else:
                self.vertexes = md.vertexes(indexed='faces')
                if self.opts['computeNormals']:
                    if self.opts['smooth']:
                        self.normals = md.vertexNormals(indexed='faces')
                    else:
                        self.normals = md.faceNormals(indexed='faces')
                self.faces = None
                if md.hasVertexColor():
                    self.colors = md.vertexColors(indexed='faces')
                elif md.hasFaceColor():
                    self.colors = md.faceColors(indexed='faces')
                    
            if self.opts['drawEdges']:
                if not md.hasFaceIndexedData():
                    self.edges = md.edges()
                    self.edgeVerts = md.vertexes()
                else:
                    self.edges = md.edges()
                    self.edgeVerts = md.vertexes(indexed='faces')
            return
    
    def paint(self):
        self.setupGLState()
        
        self.parseMeshData()        

        

        if self.opts['drawFaces']:
            with self.shader():
                verts = self.vertexes
                norms = self.normals
                color = self.colors
                faces = self.faces
                textures = self.textures
                if verts is None:
                    return
                glEnableClientState(GL_VERTEX_ARRAY)

                try:
                    glVertexPointerf(verts)

                    if textures is None:
                        if self.colors is None:
                            color = self.opts['color']
                            if isinstance(color, QtGui.QColor):
                                glColor4f(*color.getRgbF())
                            else:
                                glColor4f(*color)
                        else:
                            glEnableClientState(GL_COLOR_ARRAY)
                            glColorPointerf(color)
                    else:
                        mtl = self.textures["mtl"][self.textures['name']]

                        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                        scale = np.array(mtl['map_Kd']['scale'])
                        offset = np.array(mtl['map_Kd']['offset'])
                        t = np.array(self.textures["coords"]) / scale[:2] - offset[:2]/scale[:2]

                        glTexCoordPointerf(t)
                        glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])

                    if norms is not None:
                        glEnableClientState(GL_NORMAL_ARRAY)
                        glNormalPointerf(norms)
                    
                    if faces is None:
                        glDrawArrays(GL_TRIANGLES, 0, np.product(verts.shape[:-1]))
                    else:
                        faces = faces.astype(np.uint32).flatten()
                        glDrawElements(GL_TRIANGLES, faces.shape[0], GL_UNSIGNED_INT, faces)
                finally:
                    glDisableClientState(GL_NORMAL_ARRAY)
                    glDisableClientState(GL_VERTEX_ARRAY)
                    glDisableClientState(GL_COLOR_ARRAY)
                    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                    glDisable(GL_TEXTURE_2D)
                    glDisable(GL_TEXTURE_GEN_S)
                    glDisable(GL_TEXTURE_GEN_T)
                    glBindTexture(GL_TEXTURE_2D, 0)

           
        if self.opts['drawEdges']:
            verts = self.edgeVerts
            edges = self.edges
            glEnableClientState(GL_VERTEX_ARRAY)
            try:
                glVertexPointerf(verts)
                
                if self.edgeColors is None:
                    color = self.opts['edgeColor']
                    if isinstance(color, QtGui.QColor):
                        glColor4f(*color.getRgbF())
                    else:
                        glColor4f(*color)
                else:
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointerf(color)
                edges = edges.flatten()
                glDrawElements(GL_LINES, edges.shape[0], GL_UNSIGNED_INT, edges)
            finally:
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)

if __name__ == '__main__':
    content = MTL("DevEv/metadata/RoomData/scene/Room.mtl")
    print(content.keys())
