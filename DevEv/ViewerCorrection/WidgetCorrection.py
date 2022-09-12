from PyQt5.QtWidgets import QStyle, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QListWidget, \
                                QAbstractItemView, QInputDialog, QListWidgetItem
from PyQt5.QtCore import pyqtSignal, QDir, Qt
#from PyQt5.QtGui import QMessageBox
import pyqtgraph as pg
import cv2
import pkg_resources
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from scipy import interpolate

from .GaussianProcess import get_uncertainty

class ListWidgetItem(QListWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except Exception:
            return QListWidgetItem.__lt__(self, other)


def rotation_matrix_from_vectors(a, b):
    c = np.dot(a, b)
    if c == 1.0: return np.eye(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def read_cameras(filename):
    cams = np.load(filename, allow_pickle = True).item()
    return cams

def project_2d(poses, cams, h, w):
    hh, ww = h//2, w//2

    p3d = poses["pos"]
    p3d_head = poses["att"]
    p2d_list = {}
    for c, cam in cams.items():
        t = -cam["R"] @ cam["T"]
        p2d, _ = cv2.projectPoints(np.array([p3d, p3d_head]).T, cam["r"], t, cam["mtx"], cam["dist"])
        p2d = p2d.reshape(-1,2)
        if not (0 < p2d[0,0] < ww and 0 < p2d[0,1] < hh): continue
        if not (0 < p2d[1,0] < ww and 0 < p2d[1,1] < hh): continue

        if c%4 == 1: p2d[:,0] += ww
        elif c%4 == 2: p2d[:,1] += hh
        elif c%4 == 3:  p2d += np.array([ww, hh])
        p2d_list[c%4] = {}
        #if 0 < p2d[0,0] < w and 0 < p2d[0,1] < h:
        p2d_list[c%4]["head"] = p2d[0].astype("int")
        #if 0 < p2d[1,0] < w and 0 < p2d[1,1] < h:
        p2d_list[c%4]["att"] = p2d[1].astype("int")
    return p2d_list

class CorrectionWindow(QWidget):
    frame_id = pyqtSignal(int)
    pose2d = pyqtSignal(dict)
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, viewer3D):
        super().__init__()

        self.setWindowTitle("Correction Tool") 
        self.resize(400 , 200 )

        self.viewer3D = viewer3D
        self.frame_list = sorted(list(range(0, 300, 20)))#[0, 100, 200, 300]
        self.curr_indice = 0
        self.corrected_list = []
        ## Init cameras
        self.setHW(0, 0)

        self.frame_listW = QListWidget()
        self.frame_listW.setSelectionMode(QAbstractItemView.SingleSelection)
        for f in self.frame_list:
            self.frame_listW.addItem(ListWidgetItem(str(f)))
        self.frame_listW.itemDoubleClicked.connect(self.select_frame)
        self.frame_listW.setCurrentRow(self.curr_indice)
        self.frame_listW.setSortingEnabled(True)
        ## Some text
        self.framelabel = QLabel("Frame: " + str(self.frame_list[self.curr_indice]))
        #self.nextframelabel = QLabel("Next Frame: " + str(self.frame_list[(self.curr_indice + 1) % len(self.frame_list)]))

        self.addButton = QPushButton("&+")
        self.addButton.setEnabled(True)
        self.addButton.setStatusTip('add a frame for correction')
        self.addButton.clicked.connect(self.add_frame)

        self.removeButton = QPushButton("&-")
        self.removeButton.setEnabled(True)
        self.removeButton.setStatusTip('remove a frame from correction')
        self.removeButton.clicked.connect(self.remove_frame)

        ## Button
        self.prevframeButton = QPushButton("&Previous")
        self.prevframeButton.setEnabled(True)
        self.prevframeButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.prevframeButton.setShortcut("Left")
        self.prevframeButton.setStatusTip('Go to previous frame')
        self.prevframeButton.clicked.connect(self.prev_frame)

        self.frameButton = QPushButton("&Refresh")
        self.frameButton.setEnabled(True)
        self.frameButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.frameButton.clicked.connect(self.update_frame)

        self.nextframeButton = QPushButton("&Next")
        self.nextframeButton.setEnabled(True)
        self.nextframeButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.nextframeButton.setShortcut("Right")
        self.nextframeButton.setStatusTip('Go to next frame')
        self.nextframeButton.clicked.connect(self.next_frame)

        self.saveButton = QPushButton("&Save")
        self.saveButton.setEnabled(True)
        self.saveButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.saveButton.clicked.connect(self.save_pos)

        ## Project 2d button
        self.project2dButton = QPushButton("&Project 2D")
        self.project2dButton.setEnabled(True)
        self.project2dButton.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        self.project2dButton.clicked.connect(self.project2D)

        ## Run Assistant button 
        self.runGPButton = QPushButton("&Run Assistant")
        self.runGPButton.setEnabled(True)
        self.runGPButton.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.runGPButton.clicked.connect(self.runGP)

        ## RFinish button 
        self.finishButton = QPushButton("&Save and Finish")
        self.finishButton.setEnabled(True)
        self.finishButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.finishButton.setShortcut("Ctrl+S")
        self.finishButton.setStatusTip('Save File')
        self.finishButton.clicked.connect(self.finish)

        ## Input editing
        self.max_XEdit = pg.SpinBox()
        self.max_XEdit.setDecimals(3)
        self.max_XEdit.setRange(-15.0, 15.0)
        self.max_XEdit.setMinimumSize(80,30)
        self.max_XEdit.setSingleStep(0.01)
        self.max_XEdit.sigValueChanging.connect(self.x_changed)
        #self.max_XEdit.editingFinished.connect(self.save_pos)

        self.max_YEdit = pg.SpinBox()
        self.max_YEdit.setDecimals(3)
        self.max_YEdit.setRange(-20.0, 20.0)
        self.max_YEdit.setMinimumSize(80,30)
        self.max_YEdit.setSingleStep(0.01)
        self.max_YEdit.sigValueChanging.connect(self.y_changed)
        #self.max_YEdit.editingFinished.connect(self.save_pos)

        self.max_ZEdit = pg.SpinBox()
        self.max_ZEdit.setDecimals(3)
        self.max_ZEdit.setRange(0.0, 10.0)
        self.max_ZEdit.setMinimumSize(80,30)
        self.max_ZEdit.setSingleStep(0.01)
        self.max_ZEdit.sigValueChanging.connect(self.z_changed)
        #self.max_ZEdit.editingFinished.connect(self.save_pos)

        self.max_YawEdit = pg.SpinBox()
        self.max_YawEdit.setDecimals(3)
        self.max_YawEdit.setRange(0, 360)
        self.max_YawEdit.setMinimumSize(80,30)
        self.max_YawEdit.setSingleStep(1)
        self.max_YawEdit.setWrapping(True)
        self.max_YawEdit.sigValueChanging.connect(self.yaw_changed)
        #self.max_YawEdit.editingFinished.connect(self.save_pos)

        self.max_PitchEdit = pg.SpinBox()
        self.max_PitchEdit.setDecimals(3)
        self.max_PitchEdit.setRange(0, 360)
        self.max_PitchEdit.setMinimumSize(80,30)
        self.max_PitchEdit.setSingleStep(1)
        self.max_PitchEdit.setWrapping(True)
        self.max_PitchEdit.sigValueChanging.connect(self.pitch_changed)
        #self.max_PitchEdit.editingFinished.connect(self.save_pos)

        self.max_RollEdit = pg.SpinBox()
        self.max_RollEdit.setDecimals(3)
        self.max_RollEdit.setRange(0, 360)
        self.max_RollEdit.setMinimumSize(80,30)
        self.max_RollEdit.setSingleStep(1)
        self.max_RollEdit.setWrapping(True)
        self.max_RollEdit.sigValueChanging.connect(self.roll_changed)
        #self.max_ZEdit.editingFinished.connect(self.save_pos)

        # Label
        self.xlabel = QLabel("X")
        self.xlabel.setBuddy(self.max_XEdit)
        self.ylabel = QLabel("Y")
        self.ylabel.setBuddy(self.max_YEdit)
        self.zlabel = QLabel("Z")
        self.zlabel.setBuddy(self.max_ZEdit)

        self.yawlabel = QLabel("Yaw")
        self.yawlabel.setBuddy(self.max_YawEdit)
        self.pitchlabel = QLabel("Pitch")
        self.pitchlabel.setBuddy(self.max_PitchEdit)
        self.rolllabel = QLabel("Roll")
        self.rolllabel.setBuddy(self.max_RollEdit)

        # Layout
        layoutButton = QHBoxLayout()
        layoutButton.addWidget(self.prevframeButton)
        layoutButton.addWidget(self.frameButton)
        layoutButton.addWidget(self.nextframeButton)


        layoutFrameList = QVBoxLayout()
        layoutFrameList.addWidget(self.framelabel)
        layoutFrameList.addWidget(self.addButton)
        layoutFrameList.addWidget(self.removeButton)

        layoutInfo = QHBoxLayout()
        layoutInfo.addWidget(self.frame_listW)
        layoutInfo.addLayout(layoutFrameList)

        layoutLeft = QVBoxLayout()
        layoutLeft.addLayout(layoutButton)    
        layoutLeft.addLayout(layoutInfo)    
        layoutLeft.addWidget(self.saveButton)

        layoutPos = QHBoxLayout()
        layoutPos.addWidget(self.xlabel)
        layoutPos.addWidget(self.max_XEdit)
        layoutPos.addWidget(self.ylabel)
        layoutPos.addWidget(self.max_YEdit)
        layoutPos.addWidget(self.zlabel)
        layoutPos.addWidget(self.max_ZEdit)

        layoutOr = QHBoxLayout()
        layoutOr.addWidget(self.yawlabel)
        layoutOr.addWidget(self.max_YawEdit)
        layoutOr.addWidget(self.pitchlabel)
        layoutOr.addWidget(self.max_PitchEdit)
        layoutOr.addWidget(self.rolllabel)
        layoutOr.addWidget(self.max_RollEdit)

        inputLayout = QVBoxLayout()
        inputLayout.addLayout(layoutPos)
        inputLayout.addLayout(layoutOr)

        featureLayout = QHBoxLayout()
        featureLayout.addWidget(self.project2dButton)
        featureLayout.addWidget(self.runGPButton)

        subLayout = QVBoxLayout()
        subLayout.addLayout(featureLayout)
        subLayout.addWidget(self.finishButton)

        mainLayout = QHBoxLayout()     
        mainLayout.addLayout(layoutLeft)
        mainLayout.addLayout(inputLayout)
        mainLayout.addLayout(subLayout)

        self.setLayout(mainLayout)
        self.update_frame()

    def setHW(self, h, w):
        if h == 0 or w == 0:
            ## Read camera parameters:
            cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_BottomLeft_trim.npy')
            self.cams = read_cameras(cam_file)
        self.h, self.w = 720, 1280
        if h == 720 and w == 1280:
            cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_BottomLeft_trim.npy')
            self.cams = read_cameras(cam_file)
        else:
            cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_MobileInfants_trim.npy')
            self.cams = read_cameras(cam_file)
            self.h, self.w = 1080, 1920
        return

    def select_frame(self, item):
        self.curr_indice = self.frame_listW.currentRow()
        self.update_frame()
        return

    def add_frame(self):
        value, ok = QInputDialog.getInt(self, 'Add frame', 'Enter a frame number to add \n(single entry)')
        if value in self.frame_list:
            print("Frame already selected")
            return
        if value not in self.viewer3D.attention:
            print("Frame does not have attention")
            return            
        if ok:
            self.frame_list.append(value)
            self.frame_list = sorted(self.frame_list)

            self.frame_listW.addItem(ListWidgetItem(str(value)))
            self.curr_indice = self.frame_listW.currentRow()
            if len(self.frame_list) == 1: 
                self.frame_listW.setCurrentRow(0)
                self.curr_indice = 0
                self.update_frame()

    def remove_frame(self):
        if len(self.frame_list) == 0: 
            return
        f = self.frame_list[self.curr_indice]
        del self.frame_list[self.curr_indice]
        self.frame_listW.takeItem(self.curr_indice)
        self.curr_indice = self.frame_listW.currentRow()
        if f in self.corrected_list:
            self.corrected_list.remove(f)
        if len(self.frame_list) == 0: 
            self.curr_indice = -1
            self.framelabel.setText("No Frames")
        self.update_frame()

    def update_info(self):
        curr_frame = self.frame_list[self.curr_indice]
        if not curr_frame in self.viewer3D.attention:
            return
        data = self.viewer3D.attention[curr_frame]
        pos = data["head"]
        self.old_x, self.old_y, self.old_z = pos[0], pos[1], pos[2]
        self.max_XEdit.setValue(pos[0])
        self.max_YEdit.setValue(pos[1])
        self.max_ZEdit.setValue(pos[2])

        self.origin_vec = (data["u"][1] - data["u"][0])/np.linalg.norm(data["u"][1] - data["u"][0])
        m = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), self.origin_vec)
        angle = np.rint(R.from_matrix(m).as_euler("xyz", degrees=True)) % 360

        self.origin_angles = angle
        self.old_yaw, self.old_pitch, self.old_roll = angle[0], angle[1], angle[2]
        self.max_YawEdit.setValue(angle[0])
        self.max_PitchEdit.setValue(angle[1])
        self.max_RollEdit.setValue(angle[2])
        return

    def update_frame(self):
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        self.frame_id.emit(curr_frame)
        self.framelabel.setText("Frame: " + str(curr_frame))
        self.update_info()
        self.pose2d.emit({})
        return

    def next_frame(self):
        if self.curr_indice == -1: return
        self.curr_indice = (self.curr_indice + 1) % len(self.frame_list) 
        self.frame_listW.setCurrentRow(self.curr_indice)
        self.update_frame()
        #self.pose2d.emit({})
        return

    def prev_frame(self):
        if self.curr_indice == -1: return
        self.curr_indice = (self.curr_indice - 1) % len(self.frame_list) 
        self.frame_listW.setCurrentRow(self.curr_indice)
        self.update_frame()
        return

    def x_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        self.viewer3D.translate_attention(curr_frame, value - self.old_x, 0.0, 0.0)
        self.old_x = value
        return

    def y_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        self.viewer3D.translate_attention(curr_frame, 0.0, value - self.old_y, 0.0)
        self.old_y = value
        return

    def z_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        self.viewer3D.translate_attention(curr_frame, 0.0, 0.0, value - self.old_z)
        self.old_z = value
        return

    def yaw_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        origin = np.array([self.old_x, self.old_y, self.old_z])
        self.viewer3D.rotate_attention(curr_frame, value - self.old_yaw, 1.0, 0.0, 0.0, origin)
        self.old_yaw = value
        return

    def pitch_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        origin = np.array([self.old_x, self.old_y, self.old_z])
        self.viewer3D.rotate_attention(curr_frame, value - self.old_pitch, 0.0, 1.0, 0.0, origin)
        self.old_pitch = value
        return

    def roll_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        origin = np.array([self.old_x, self.old_y, self.old_z])
        self.viewer3D.rotate_attention(curr_frame, value - self.old_roll, 0.0, 0.0, 1.0, origin)
        self.old_roll = value
        return

    def save_pos(self):
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        x, y, z = self.max_XEdit.value(), self.max_YEdit.value(), self.max_ZEdit.value()
        x_, y_, z_ = self.max_YawEdit.value(), self.max_PitchEdit.value(), self.max_RollEdit.value()
        pos = np.array([x,y,z])

        m = R.from_euler("xyz", np.array([x_, y_, z_]) - self.origin_angles, degrees=True).as_matrix() 
        self.origin_vec = np.dot(m, self.origin_vec)
        self.origin_vec = self.origin_vec / np.linalg.norm(self.origin_vec)

        self.viewer3D.modify_attention(curr_frame, pos, self.origin_vec)
        self.origin_angles = np.array([x_, y_, z_])
        curr_frame = self.frame_list[self.curr_indice]
        self.frame_id.emit(curr_frame)
        if curr_frame not in self.corrected_list:
            self.corrected_list.append(curr_frame)
        self.frame_listW.item(self.curr_indice).setBackground(Qt.green)
        return

    def project2D(self):    
        x, y, z = self.max_XEdit.value(), self.max_YEdit.value(), self.max_ZEdit.value()
        x_, y_, z_ = self.max_YawEdit.value(), self.max_PitchEdit.value(), self.max_RollEdit.value()
        pos = np.array([x,y,z])

        m = R.from_euler("xyz", np.array([x_, y_, z_]) - self.origin_angles, degrees=True).as_matrix() 
        vec = np.dot(m, self.origin_vec)
        vec = vec / np.linalg.norm(vec)
        att = self.viewer3D.collision(pos, vec)

        p = {"pos":pos, "att":att}
        poses = project_2d(p, self.cams, self.h, self.w)
        self.pose2d.emit(poses)
        return

    def runGP(self):
        x_tr = []
        for f, p in self.viewer3D.attention.items():
            p, v = p["u"][0], p["u"][1]-p["u"][0]
            v = v/np.linalg.norm(v)
            info = np.concatenate([p, v], axis=0)
            x_tr.append(info)
        x_tr = np.array(x_tr)

        if len(self.corrected_list) > 0:
            mask = build_mask(self.corrected_list, len(x_tr))[:, np.newaxis]
            #plt.plot(mask)
            #plt.show()
            self.corrected_list = [0] + self.corrected_list + [len(x_tr)-1]
            correction = x_tr[self.corrected_list]
            f = interpolate.interp1d(self.corrected_list, correction, axis=0)
            x_interp = f(np.arange(0, len(x_tr), 1))
            x_tr = (1-mask)*x_tr + mask * x_interp
            #plt.plot(x_tr, "-r")
            #plt.plot(x_interp, "-b")
            #plt.plot(x_tr, "-g")
            #plt.show()
        self.write_attention()
        self.frame_list = sorted(get_uncertainty(x_tr))
        if len(self.frame_list == 0):
            self.curr_indice = -1
            self.frame_listW.clear()
            return

        self.curr_indice = 0
        self.frame_listW.clear()
        self.frame_listW.addItems([str(f) for f in self.frame_list])
        self.frame_listW.setCurrentRow(0)
        self.corrected_list = []
        self.update_frame()
        return

    def write_attention(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Corrected Results", QDir.homePath() + "/corrected.txt", "Text files (*.txt)")
        if fileName == '':
            return
        with open(fileName, "w") as w:
            w.write("")
            for f, p in self.viewer3D.attention.items():
                p, v = p["u"][0], p["u"][1]-p["u"][0]
                v = v/np.linalg.norm(v)
                att = self.viewer3D.collision(p, v)
                w.write("{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    f, p[0], p[1], p[2], v[0], v[1], v[2], att[0], att[1], att[2]
                ))
        self.viewer3D.read_attention(fileName)
        return

    def finish(self):
        self.write_attention()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you want to Quit the Correction Tool?")
        msg.setWindowTitle("Quit")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
        retval = msg.exec_()
        if retval == QMessageBox.Yes:
            print("Pressed", QMessageBox.Yes)
            self.close()
        else:
            print("Pressed", QMessageBox.No)


def build_mask(frames, N, sigma = 1, threshold = 30):
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
        end_l = threshold*2 - abs(max(0, frames[i+1]+threshold - N))
        mask[start:end] = linear[start_l:end_l] 

    return mask

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))