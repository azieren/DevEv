import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os
import cv2
import sys
import json
import time
from matplotlib import cm
from scipy.spatial import ConvexHull

def get_balloon():
    md = gl.MeshData.sphere(rows=6, cols=6, radius=1)

    m1 = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(245/255.0,  148/255.0, 241/255.0, 0.5),
        shader="balloon",
        glOptions='translucent',
    )

    m1.translate(-5.0, 2.5, 1.0)
    return m1

def get_red_ball():
    md = gl.MeshData.sphere(rows=6, cols=6, radius=0.25)

    m1 = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(242/255.0,  92/255.0, 46/255.0, 0.5),
        shader="balloon",
        glOptions='translucent',
    )

    m1.translate(0.5, 0.5, 0.25)
    return m1

def get_ring():
    md = gl.MeshData.sphere(rows=6, cols=6, radius=0.25)

    m1 = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(0.8, 0.8, 0.0, 0.5),
        shader="balloon",
        glOptions='translucent',
    )
    
    m1.scale(1.0,1.0,0.2)
    m1.translate(5.1, 6.2, 0.25-0.2)
    
    return m1

def get_piggy():
    md = gl.MeshData.sphere(rows=6, cols=6, radius=0.3)

    m1 = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(245/255.0,  148/255.0, 241/255.0, 0.5),
        shader="balloon",
        glOptions='translucent',
    )
    
    m1.translate(-1.5, -14.0, 0.3)
    
    return m1

def get_shovel():
    c = (1,1,1,0.4)
    faces = np.array([[0,1,2], [0,2,3], [0,4,5], [0,3,2], [0,1,4], [3,2,5], [1,2,5], [4,1,5] , [6,7,8] , [8,7,9]])
    p = np.array([[0.0, 0.0, 0.0], 
                    [0.0, 0.62, 0.0], 
                    [0.8, 0.62, 0.0],
                    [0.8, 0.0, 0.0],
                    [0.0, 0.62, 0.2],
                    [0.8, 0.62, 0.2],

                    [0.34, 0.62, 0.2],
                    [0.34, 0.62 + 0.35, 0.2],
                    [0.46, 0.62, 0.2],
                    [0.46, 0.62 + 0.35, 0.2],
                    ])
    d = gl.MeshData(vertexes=p, faces=faces)
    plane = gl.GLMeshItem(meshdata=d, color = c, shader='viewNormalColor', glOptions='translucent')
    
    plane.rotate(-90.0, 0.0, 0.0, 1.0)
    plane.rotate(-60.0, 0.0, 1.0, 0.0)
    plane.translate(-0.3, -0.1, 0.0)
    return plane

def get_farm():
    c = (1.0,0.0,0.0,0.4)
    p = np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1, 0], #2
                     [0, 0, 1], #3
                     [1, 1, 0], #4
                     [1, 1, 1], #5
                     [0, 1, 1], #6
                     [1, 0, 1]], dtype="float")#7

    faces = np.array([[1,7,0], [1,3,7],
                  [1,4,2], [1,0,4],
                  [1,6,2], [1,3,6],
                  [0,5,4], [0,7,5],
                  [2,5,4], [2,6,5],
                  [3,5,6], [3,7,5]])                     
    d = gl.MeshData(vertexes=p, faces=faces)

    plane = gl.GLMeshItem(meshdata=d, color = c, shader='viewNormalColor', glOptions='translucent')
    plane.translate(0.5, -1.8, 3.5)
    plane.scale(1.5, 1.1, 0.25)
    return plane

def get_xyl():
    c = (0.0,1.0,0.0,0.4)
    p = np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1, 0], #2
                     [0, 0, 1], #3
                     [1, 1, 0], #4
                     [1, 1, 1], #5
                     [0, 1, 1], #6
                     [1, 0, 1]], dtype="float")#7

    faces = np.array([[1,7,0], [1,3,7],
                  [1,4,2], [1,0,4],
                  [1,6,2], [1,3,6],
                  [0,5,4], [0,7,5],
                  [2,5,4], [2,6,5],
                  [3,5,6], [3,7,5]])                     
    d = gl.MeshData(vertexes=p, faces=faces)

    plane = gl.GLMeshItem(meshdata=d, color = c, glOptions='translucent')
    plane.scale(2.0, 1.0, 0.5)
    plane.rotate(-30, 0.0, 0.0, 1.0)
    plane.translate(6.5, 1.5, 0.0)
    
    return plane

def get_red_toy():
    c = (1.0,0.0,0.0,0.4)
    p = np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1, 0], #2
                     [0, 0, 1], #3
                     [1, 1, 0], #4
                     [1, 1, 1], #5
                     [0, 1, 1], #6
                     [1, 0, 1]], dtype="float")#7

    faces = np.array([[1,7,0], [1,3,7],
                  [1,4,2], [1,0,4],
                  [1,6,2], [1,3,6],
                  [0,5,4], [0,7,5],
                  [2,5,4], [2,6,5],
                  [3,5,6], [3,7,5]])                     
    d = gl.MeshData(vertexes=p, faces=faces)

    plane = gl.GLMeshItem(meshdata=d, color = c, shader='viewNormalColor', glOptions='translucent')
    plane.scale(1.0, 1.0, 1.5)
    #plane.rotate(-30, 0.0, 0.0, 1.0)
    plane.translate(3.0, -14.0, 0.0)
    
    return plane

def get_tree():
    c = (1.0,0.0,0.0,0.4)
    p = np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1, 0], #2
                     [0, 0, 1], #3
                     [1, 1, 0], #4
                     [1, 1, 1], #5
                     [0, 1, 1], #6
                     [1, 0, 1]], dtype="float")#7

    faces = np.array([[1,7,0], [1,3,7],
                  [1,4,2], [1,0,4],
                  [1,6,2], [1,3,6],
                  [0,5,4], [0,7,5],
                  [2,5,4], [2,6,5],
                  [3,5,6], [3,7,5]])                     
    d = gl.MeshData(vertexes=p, faces=faces)

    plane = gl.GLMeshItem(meshdata=d, color = c, shader='viewNormalColor', glOptions='translucent')
    plane.scale(1.0, 1.0, 2.4)
    #plane.rotate(-30, 0.0, 0.0, 1.0)
    plane.translate(-7, -14.0, 0.0)
    
    return plane

"""def draw_planes():
    self.planes = {}
    self.room_line = []
    c = (1,1,1,0.4)
    print(len(self.plane_list))
    for i,p in enumerate(self.plane_list):       
        #if 101 > i > 98: 
        if i == 99 or i == 100:
            # Do not show floor and ceiling
            continue
        faces = np.array([[0,1,2], [0,2,3]])
        d = gl.MeshData(vertexes=p, faces=faces)
        plane = gl.GLMeshItem(meshdata=d, color = c, shader='viewNormalColor', glOptions='translucent')
        self.planes[i] = plane
        self.addItem(plane)
        for j in range(3):
            line_points = np.array([p[j], p[j+1]])               
            l = gl.GLLinePlotItem(pos = line_points, color = (1.0, 1.0, 1.0, 0.7), width= 2.5, glOptions='translucent')
            self.room_line.append(l)
            self.addItem(l)

        line_points = np.array([p[-1], p[0]])
        l = gl.GLLinePlotItem(pos = line_points, color = (1.0, 1.0, 1.0, 0.7), width= 2.5, glOptions='translucent')
        self.room_line.append(l)
        self.addItem(l)"""

#if __name__ == "__main__":
    



