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
from .utils import rotation_matrix_from_vectors, project_2d, build_mask, to_3D

class ListWidgetItem(QListWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except Exception:
            return QListWidgetItem.__lt__(self, other)

def read_cameras(filename):
    cams = np.load(filename, allow_pickle = True).item()
    return cams

class CorrectionWindow(QWidget):
    frame_id = pyqtSignal(int)
    pose2d = pyqtSignal(dict)
    open_id = pyqtSignal(bool)
    project3d_clicked = pyqtSignal(bool)
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
        self.old_yaw, self.old_pitch, self.old_roll = 0, 0, 0
        self.old_x, self.old_y, self.old_z = 0, 0, 0
        self.old_x_att, self.old_y_att, self.old_z_att = 0, 0, 0
        self.corrected_list = []
        self.modify_att = True
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

        ## Project 3d button
        self.project3dButton = QPushButton("&Project 3D")
        self.project3dButton.setEnabled(True)
        self.project3dButton.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))

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


        ## Input editing
        self.max_XAEdit = pg.SpinBox()
        self.max_XAEdit.setDecimals(3)
        self.max_XAEdit.setRange(-15.0, 15.0)
        self.max_XAEdit.setMinimumSize(80,30)
        self.max_XAEdit.setSingleStep(0.01)
        self.max_XAEdit.sigValueChanging.connect(self.x_att_changed)
        #self.max_XEdit.editingFinished.connect(self.save_pos)

        self.max_YAEdit = pg.SpinBox()
        self.max_YAEdit.setDecimals(3)
        self.max_YAEdit.setRange(-20.0, 20.0)
        self.max_YAEdit.setMinimumSize(80,30)
        self.max_YAEdit.setSingleStep(0.01)
        self.max_YAEdit.sigValueChanging.connect(self.y_att_changed)
        #self.max_YEdit.editingFinished.connect(self.save_pos)

        self.max_ZAEdit = pg.SpinBox()
        self.max_ZAEdit.setDecimals(3)
        self.max_ZAEdit.setRange(0.0, 20.0)
        self.max_ZAEdit.setMinimumSize(80,30)
        self.max_ZAEdit.setSingleStep(0.01)
        self.max_ZAEdit.sigValueChanging.connect(self.z_att_changed)
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

        self.x_att_label = QLabel("X Att")
        self.x_att_label.setBuddy(self.max_XAEdit)
        self.y_att_label = QLabel("Y Att")
        self.y_att_label.setBuddy(self.max_YAEdit)
        self.z_att_label = QLabel("Z Att")
        self.z_att_label.setBuddy(self.max_ZAEdit)


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

        layoutAtt = QHBoxLayout()
        layoutAtt.addWidget(self.x_att_label)
        layoutAtt.addWidget(self.max_XAEdit)
        layoutAtt.addWidget(self.y_att_label)
        layoutAtt.addWidget(self.max_YAEdit)
        layoutAtt.addWidget(self.z_att_label)
        layoutAtt.addWidget(self.max_ZAEdit)

        inputLayout = QVBoxLayout()
        inputLayout.addLayout(layoutPos)
        inputLayout.addLayout(layoutOr)
        inputLayout.addLayout(layoutAtt)

        featureLayout = QHBoxLayout()
        featureLayout.addWidget(self.project2dButton)
        featureLayout.addWidget(self.runGPButton)

        subLayout = QVBoxLayout()
        subLayout.addLayout(featureLayout)
        subLayout.addWidget(self.project3dButton)
        subLayout.addWidget(self.finishButton)

        mainLayout = QHBoxLayout()     
        mainLayout.addLayout(layoutLeft)
        mainLayout.addLayout(inputLayout)
        mainLayout.addLayout(subLayout)

        self.setLayout(mainLayout)
        self.update_frame()

    def setHW(self, h, w):
        cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/cameras.npy')
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

        pos_A = data["att"]
        self.old_x_att, self.old_y_att, self.old_z_att = pos_A[0], pos_A[1], pos_A[2]
        self.max_XAEdit.setValue(pos_A[0])
        self.max_YAEdit.setValue(pos_A[1])
        self.max_ZAEdit.setValue(pos_A[2])

        self.origin_vec = (data["u"][1] - data["u"][0])/np.linalg.norm(data["u"][1] - data["u"][0])
        self.change_att_direction(self.origin_vec)

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
        return

    def prev_frame(self):
        if self.curr_indice == -1: return
        self.curr_indice = (self.curr_indice - 1) % len(self.frame_list) 
        self.frame_listW.setCurrentRow(self.curr_indice)
        self.update_frame()
        return

    def x_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention(curr_frame, value - self.old_x, 0.0, 0.0)
        self.old_x = value
        self.modify_att = False
        self.change_att_direction(new_vec)
        self.modify_att = True
        return

    def y_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention(curr_frame, 0.0, value - self.old_y, 0.0)
        self.old_y = value
        self.modify_att = False
        self.change_att_direction(new_vec)  
        self.modify_att = True     
        return

    def z_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention(curr_frame, 0.0, 0.0, value - self.old_z)
        self.old_z = value
        self.modify_att = False
        self.change_att_direction(new_vec)
        self.modify_att = True
        return

    def change_att_direction(self, vec):
        if type(vec) == bool and not vec : return
        self.origin_vec = vec

        m = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), vec)
        angle = np.rint(R.from_matrix(m).as_euler("xyz", degrees=True)) % 360

        self.origin_angles = angle
        self.old_yaw, self.old_pitch, self.old_roll = angle[0], angle[1], angle[2]
        self.max_YawEdit.setValue(angle[0])
        self.max_PitchEdit.setValue(angle[1])
        self.max_RollEdit.setValue(angle[2])
        return

    def x_att_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention_p(curr_frame, value - self.old_x_att, 0.0, 0.0)
        self.old_x_att = value
        self.change_att_direction(new_vec)
        return

    def y_att_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention_p(curr_frame, 0.0, value - self.old_y_att, 0.0)
        self.old_y_att = value
        self.change_att_direction(new_vec)
        return

    def z_att_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        new_vec = self.viewer3D.translate_attention_p(curr_frame, 0.0, 0.0, value - self.old_z_att)
        self.old_z_att = value
        self.change_att_direction(new_vec)
        return

    def yaw_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        att = self.viewer3D.rotate_attention(curr_frame, value - self.old_yaw, "x", self.modify_att)
        self.old_yaw = value
        self.update_att(att)
        return

    def pitch_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        att = self.viewer3D.rotate_attention(curr_frame, value - self.old_pitch, "y", self.modify_att)
        self.old_pitch = value
        self.update_att(att)
        return

    def roll_changed(self, box, value):
        curr_frame = self.frame_list[self.curr_indice]
        att = self.viewer3D.rotate_attention(curr_frame, value - self.old_roll, "z", self.modify_att)
        self.old_roll = value
        self.update_att(att)
        return

    def update_att(self, att):
        if type(att) == bool or not self.modify_att: return
        self.old_x_att, self.old_y_att, self.old_z_att = att[0], att[1], att[2]
        self.max_XAEdit.setValue(att[0])
        self.max_YAEdit.setValue(att[1])
        self.max_ZAEdit.setValue(att[2])
        return

    def save_pos(self):
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        """x, y, z = self.max_XEdit.value(), self.max_YEdit.value(), self.max_ZEdit.value()
        x_, y_, z_ = self.max_YawEdit.value(), self.max_PitchEdit.value(), self.max_RollEdit.value()
        m = R.from_euler("xyz", np.array([x_, y_, z_]) - self.origin_angles, degrees=True).as_matrix() 
        self.origin_vec = np.dot(m, self.origin_vec)
        self.origin_vec = self.origin_vec / np.linalg.norm(self.origin_vec)"""

        att = self.viewer3D.modify_attention(curr_frame)
        self.update_att(att)
        #self.origin_angles = np.array([x_, y_, z_])
        curr_frame = self.frame_list[self.curr_indice]
        self.frame_id.emit(curr_frame)
        if curr_frame not in self.corrected_list:
            self.corrected_list.append(curr_frame)
        self.frame_listW.item(self.curr_indice).setBackground(Qt.green)
        return

    def project3D(self, data):
        if len(data) < 2: return
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        item = self.viewer3D.drawn_item[curr_frame]

        att = to_3D(data, self.cams, self.h, self.w)
        u = att - item["head"].pos
        u = u / np.linalg.norm(u)
        att = self.viewer3D.collision(item["head"].pos, u)
 
        p = {"pos":item["head"].pos, "att":att}
        poses = project_2d(p, self.cams, self.h, self.w)

        new_vec = self.viewer3D.translate_attention_p(curr_frame, 
                        att[0] - self.old_x_att, att[1] - self.old_y_att, att[2] - self.old_z_att)
        self.old_x_att,  self.old_y_att,  self.old_z_att = att[0], att[1], att[2]
        self.change_att_direction(new_vec)

        self.pose2d.emit(poses)
        return

    def project2D(self):    
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        item = self.viewer3D.drawn_item[curr_frame]
        pos = item["head"].pos
        u = item["att"].pos - pos
        u = u / np.linalg.norm(u)
        att = self.viewer3D.collision(pos, u)

        p = {"pos":pos, "att":att}
        poses = project_2d(p, self.cams, self.h, self.w)
        self.pose2d.emit(poses)
        return

    def propagate(self):
        x_tr, frame_list, corrected_list = [], [], []
        for i, (f, p) in enumerate(self.viewer3D.attention.items()):
            p, v = p["u"][0], p["u"][1]-p["u"][0]
            v = v/np.linalg.norm(v)
            info = np.concatenate([p, v], axis=0)
            x_tr.append(info)
            frame_list.append(f)
            if f in self.corrected_list:
                corrected_list.append(i)
        x_tr = np.array(x_tr)

        if len(self.corrected_list) == 0: return x_tr, frame_list

        mask = build_mask(corrected_list, len(x_tr))[:, np.newaxis]
        #plt.plot(mask[:100])
        #plt.show()
        if 0 not in corrected_list: corrected_list = [0] + corrected_list
        if len(x_tr)-1 not in corrected_list: corrected_list = corrected_list + [len(x_tr)-1]
        correction = x_tr[corrected_list]

        f = interpolate.interp1d(corrected_list, correction, axis=0)
        x_interp = f(np.arange(0, len(x_tr), 1))
        #plt.plot(x_tr[:,0], "-r")
        x_tr = (1-mask)*x_tr + mask * x_interp
        #plt.plot(x_interp[:,0], "-b")
        #plt.plot(x_tr[:,0], "-g")
        #plt.show()

        for i, f in enumerate(frame_list):
            p = self.viewer3D.attention[f]
            pos = x_tr[i, :3]
            v = x_tr[i, 3:]
            v = v/np.linalg.norm(v)
            att = self.viewer3D.collision(pos, v)
            if att is None or v is None or pos is None:
                continue
            p["head"] = pos
            p["u"][0], p["line"][0] = pos, pos
            p["u"][0] = pos + v
            p["line"][1] = att
            p["att"] = att
        return x_tr, frame_list

    def runGP(self):

        x_tr, frame_list = self.propagate()
        self.write_attention("temp.txt")
        N = len(self.viewer3D.attention) // 1800
        uncertain_frames = get_uncertainty(x_tr, max_n= N * 10)
        self.frame_list = sorted([frame_list[f] for f in uncertain_frames])
        print(self.frame_list)
        print("{} Frames proposed to correct".format(len(self.frame_list)))
        if len(self.frame_list) == 0:
            self.curr_indice = -1
            self.frame_listW.clear()
            self.corrected_list = []
            return

        self.curr_indice = 0
        self.frame_listW.clear()
        for f in self.frame_list:
            self.frame_listW.addItem(ListWidgetItem(str(f)))
        self.frame_listW.setCurrentRow(0)
        self.corrected_list = []
        self.update_frame()
        return

    def write_attention(self, fileName = None, new_att = []):
        if fileName is None:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Corrected Results", QDir.homePath() + "/corrected.txt", "Text files (*.txt)")
            if fileName == '':
                return
        with open(fileName, "w") as w:
            w.write("")
            for i, (f, p) in enumerate(self.viewer3D.attention.items()):
                pos, v = p["u"][0], p["u"][1]-p["u"][0]
                att = p["att"]
                w.write("{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    f, pos[0], pos[1], pos[2], v[0], v[1], v[2], att[0], att[1], att[2]
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
            self.close()
        return

    def closeEvent(self, event):
        self.pose2d.emit({})
        self.open_id.emit(False)
        event.accept()