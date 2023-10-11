import numpy as np
import cv2 
from scipy.spatial.transform import Rotation

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def build_mask_old(frames, N, sigma = 1, threshold = 30):
    mask = np.zeros(N)
    linear = gaussian(np.linspace(-3, 3, threshold*2), 0, sigma)

    start = max(0, frames[0]-threshold)
    end = min(N, frames[0]+threshold)
    start_l = abs(min(0, frames[0]-threshold))
    end_l = threshold*2 - abs(min(0, N - frames[0]-threshold))
    mask[start:end] = linear[start_l:end_l]
    for i in range(len(frames)-1):
        if frames[i+1] - frames[i] < threshold:
            mask[frames[i] : frames[i+1]] = 1
            continue
        start = max(0, frames[i+1]-threshold)
        end = min(N, frames[i+1]+threshold)
        start_l = abs(min(0, frames[i+1]-threshold))
        end_l = threshold*2 - abs(max(0, N - frames[i+1]-threshold))
        mask[start:end] = linear[start_l:end_l] 

    return mask

def build_mask(frames, N, sigma = 1, threshold = 30):
    mask = np.zeros(N)
    linear = gaussian(np.linspace(-3, 3, threshold*2), 0, sigma)

    for i in range(len(frames)-1):
        start, end = max(0, frames[i]-threshold), min(N, frames[i]+threshold)
        start_g, end_g = threshold - (frames[i] - start),  threshold + (end - frames[i])
        mask[start:end] = np.maximum(mask[start:end], linear[start_g:end_g])
        if frames[i+1] - frames[i] < threshold:
            mask[frames[i] : frames[i+1]] = 1
            continue

    start, end = max(0, frames[-1]-threshold), min(N, frames[-1]+threshold)
    start_g, end_g = threshold - (frames[-1] - start),  threshold + (end - frames[-1])
    mask[start:end] = np.maximum(mask[start:end], linear[start_g:end_g])
    return mask

def rotation_matrix_from_vectors(a, b):
    c = np.dot(a, b)
    if c == 1.0: return np.eye(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def project_2d(poses, cams, h, w, is_mat = False):
    hh, ww = h//4, w//2
    att_dir = None
    if "att" in poses: 
        p3d = [poses["pos"], poses["att"]]
        att_dir = poses["att"] - poses["pos"]
        att_dir = att_dir / np.linalg.norm(att_dir)
    else: p3d = [poses["pos"]]
    p2d_list = {}
    
    for c, cam in cams.items():
        has_head, has_att = True, True
        t = -cam["R"] @ cam["T"]
        p2d, _ = cv2.projectPoints(np.array(p3d).T, cam["r"], t, cam["mtx"], cam["dist"])
        p2d = p2d.reshape(-1,2)
        if not (0 < p2d[0,0] < ww and 0 < p2d[0,1] < hh): has_head = False
        if "att" in poses:
            if not (0 < p2d[1,0] < ww and 0 < p2d[1,1] < hh): has_att = False

        if c == 1: p2d[:,0] += ww
        elif c == 2: p2d[:,1] += hh
        elif c == 3:  p2d += np.array([ww, hh])
        elif c == 4:  p2d[:,1] += 2*hh
        elif c == 5:  p2d += np.array([ww, 2*hh])
        elif c == 6:  p2d[:,1] += 3*hh
        elif c == 7:  p2d += np.array([ww, 3*hh])
        p2d_list[c] = {}
        #if 0 < p2d[0,0] < w and 0 < p2d[0,1] < h:
        if has_head: p2d_list[c]["head"] = p2d[0].astype("int")
        #if 0 < p2d[1,0] < w and 0 < p2d[1,1] < h:
        if "att" in poses and has_att: p2d_list[c]["att"] = p2d[1].astype("int")
        if att_dir is not None: 
            M = rotation_matrix_from_vectors(np.array([0,0,-1.0]), att_dir)
            A  = M @ cam["R"].T
            A = Rotation.from_matrix(A).as_euler("xyz", degrees = True)
            A[0] = -A[0]
            """if is_mat and c in [0,1,2,3]:     
                M = rotation_matrix_from_vectors(np.array([0,0,1.0]), att_dir)
                A  = M @ cam["R"].T
                A = Rotation.from_matrix(A).as_euler("xyz", degrees = True)"""
               
            p2d_list[c]["angle"] = A
            
    return p2d_list

def line_intersect(pt1,u1,pt2,u2):
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    n = np.cross(u1, u2)
    n /= np.linalg.norm(n)

    n1 = np.cross(u1, n)
    n2 = np.cross(u2, n)
    
    t1 = pt1 + u1 * (np.dot((pt2 - pt1), n2) / np.dot(u1, n2))
    t2 = pt2 + u2 * (np.dot((pt1 - pt2), n1) / np.dot(u2, n1))
    p = (t1 + t2) / 2
    return p

def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors 
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p

def to_3D_old(points, cameras, h, w):
    hh, ww = h//4, w//2
    print(hh, ww)
    C = []
    P = []
    for c, info in points.items():
        if type(c) != int: continue
        x, y = info["att_p"]
        k = cameras[c]["K"]
        cam_pos = -cameras[c]["R"] @ cameras[c]["T"]

        p = [x, y, 1.0]
        X = np.dot(np.linalg.pinv(k),p)
        X = X / X[3]
        xvec = np.copy(X)
        xvec[0] = cam_pos[0]-xvec[0]
        xvec[1] = cam_pos[1]-xvec[1]
        xvec[2] = cam_pos[2]-xvec[2]
        xvec = - xvec[:3]
        P.append(xvec/np.linalg.norm(xvec))
        C.append(cam_pos)

    #att = line_intersect(C[0], P[0], C[1], P[1])
    C, P = np.array(C), np.array(P)
    att = intersect(C, C + P)[:,0]
    return att

def to_3D(points, cameras, h, w):
    hh, ww = h//4, w//2

    C = []
    P = []
    for c, info in points.items():
        if type(c) != int: continue
        P.append(np.array(info["att_p"], dtype=np.float32))
        C.append(cameras[c])
    c0, c1 = C[0], C[1]
    p0, p1 = P[0], P[1]
    p0 = cv2.undistortPoints(p0.reshape((1,1,2)), c0["mtx"], c0["dist"], None, c0["mtx"])
    p1 = cv2.undistortPoints(p1.reshape((1,1,2)), c1["mtx"], c1["dist"], None, c1["mtx"])
    p4d = cv2.triangulatePoints(c0["K"], c1["K"], p0, p1)
    p3d = (p4d[:3, :]/p4d[3, :]).T
    return p3d.reshape(3)
