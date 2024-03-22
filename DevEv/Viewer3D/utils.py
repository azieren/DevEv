import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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


def draw_cone(p0, p1, base_color, L = 2.5, n=8, R= 0.6, just_data = False):
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
    cone = gl.GLMeshItem(meshdata=d, glOptions = 'translucent', drawEdges=True, computeNormals=False, color=base_color)   
    return cone

def draw_Ncone(p0_list, p1_list, L = 2.5, n=8, R= 0.6):
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