import numpy as np
import cv2 
from scipy.spatial.transform import Rotation

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QDir

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
    # Normalize the input vectors
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = np.dot(a, b)

    if abs(c + 1.0) < 1e-6:
        # In this case, the vectors are exactly opposite, so we need a 180-degree rotation.
        # A 180-degree rotation matrix around any axis is -1 times the identity matrix.
        return -np.eye(3)

    if abs(c - 1.0) < 1e-6:
        # In this case, the vectors are already aligned, so no rotation is needed.
        return np.eye(3)

    v = np.cross(a, b)
    s = np.linalg.norm(v)

    # Skew-symmetric cross product matrix
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def project_2d(poses, cams, h, w, is_mat = False):
    hh, ww = h//4, w//2
    att_dir = None
    if "att" in poses: 
        p3d = [poses["pos"], poses["att"]]
        att_dir = poses["att"] - poses["pos"]
        att_dir = att_dir / np.linalg.norm(att_dir)
    elif "handL" in poses:
        p3d = [poses["handL"], poses["handR"]]
    else: p3d = [poses["pos"]]
    p2d_list = {}
    
    for c, cam in cams.items():
        has_head, has_att, has_handL, has_handR = False, False, False, False
        t = -cam["R"] @ cam["T"]
        p2d, _ = cv2.projectPoints(np.array(p3d).T, cam["r"], t, cam["mtx"], cam["dist"])
        p2d = p2d.reshape(-1,2)
        # Check if head is present
        if "pos" in poses and (0 < p2d[0,0] < ww and 0 < p2d[0,1] < hh): has_head = True
        # Check if Attention point is present
        if "att" in poses and (0 < p2d[1,0] < ww and 0 < p2d[1,1] < hh): has_att = True
        if "handL" in poses and (0 < p2d[0,0] < ww and 0 < p2d[0,1] < hh): has_handL = True
        if "handR" in poses and (0 < p2d[1,0] < ww and 0 < p2d[1,1] < hh): has_handR = True
        
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
        if has_att: p2d_list[c]["att"] = p2d[1].astype("int")
        if att_dir is not None: 

            M = rotation_matrix_from_vectors(np.array([0,0,1.0]), att_dir)
            A  = M @ cam["R"].T
            Cx = Rotation.from_rotvec( cam["R"].T[0] * np.radians(180)).as_matrix()
            Cy = Rotation.from_rotvec( cam["R"].T[1] * np.radians(180)).as_matrix()
            A  = A @ Cx @ Cy
            A = Rotation.from_matrix(A).as_euler("xyz",degrees = True)
                    
            p2d_list[c]["angle"] = A
        # Hands
        if has_handL: p2d_list[c]["handL"] = p2d[0].astype("int")
        if has_handR: p2d_list[c]["handR"] = p2d[1].astype("int")
            
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

def write_results(tool, source, fileName = None, is_temp = False):
    
    if fileName is None:
        fileName, _ = QFileDialog.getSaveFileName(tool, "Save Corrected Results", QDir.homePath() + "/corrected.txt", "Text files (*.txt)")
        #options=QFileDialog.DontUseNativeDialog)
        if fileName == '':
            return
    print("Writing ", tool.history_corrected)
    with open(fileName, "w") as w:
        w.write("")
        for i, (f, p) in enumerate(tool.viewer3D.attention.items()):
            pos, v = p["u"][0], p["u"][1]-p["u"][0]
            handL, handR= p["handL"], p["handR"]
            if p["att"] is not None:
                att = p["att"]
            flag = p["corrected_flag"]
            flag_h = p["corrected_flag_hand"]
            w.write("{:d},{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                f, flag, flag_h, pos[0], pos[1], pos[2], v[0], v[1], v[2], att[0], att[1], att[2], handL[0], handL[1], handL[2], handR[0], handR[1], handR[2]
            ))
            if flag_h > 0 and source == "hand": tool.history_corrected[f] = flag_h
            if flag > 0 and source == "att": tool.history_corrected[f] = flag
    if not is_temp:
        tool.viewer3D.read_attention(fileName)
    print("Corrected frames:", len([x for x, y in tool.history_corrected.items() if y == 1]))
    print("File saved at", fileName)
    
    return
