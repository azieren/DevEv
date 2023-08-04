from PyQt5.QtWidgets import QStyle, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QListWidget, \
                                QAbstractItemView, QInputDialog, QListWidgetItem
from PyQt5.QtCore import pyqtSignal, QDir, Qt
#from PyQt5.QtGui import QMessageBox
import pyqtgraph as pg
import copy
import pkg_resources
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter as filter1d

from .GaussianProcess import get_uncertainty
from .utils import rotation_matrix_from_vectors, project_2d, build_mask, to_3D

class ListWidgetItem(QListWidgetItem):
    def __lt__(self, other):
        try:
            text = self.text().split(" ")[0]
            other = other.text().split(" ")[0]
            return float(text) < float(other)
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
        self.viewer3D.attention_sig.connect(self.change_attention_sig)
        self.viewer3D.direction_sig.connect(self.change_att_direction)
        self.viewer3D.position_sig.connect(self.change_position_sig)
        self.frame_list = np.array([], dtype=int)
        self.curr_indice = -1
        self.old_yaw, self.old_pitch, self.old_roll = 0, 0, 0
        self.old_x, self.old_y, self.old_z = 0, 0, 0
        self.old_x_att, self.old_y_att, self.old_z_att = 0, 0, 0
        self.corrected_list = set()
        self.history_corrected = self.viewer3D.corrected_frames
        self.modify_att = True
        self.memory_buffer = None
        ## Init cameras
        self.setHW(0, 0)

        self.frame_listW = QListWidget()
        self.frame_listW.setSelectionMode(QAbstractItemView.SingleSelection)
        for f in self.frame_list:
            self.frame_listW.addItem(ListWidgetItem("{} - NA".format(f)))
        self.frame_listW.itemDoubleClicked.connect(self.select_frame)
        self.frame_listW.setCurrentRow(self.curr_indice)
        self.frame_listW.setSortingEnabled(True)
        ## Some text
        if len(self.frame_list) == 0: self.framelabel = QLabel("No Frame")
        else: self.framelabel = QLabel("Frame: " + str(self.frame_list[self.curr_indice]))
        #self.nextframelabel = QLabel("Next Frame: " + str(self.frame_list[(self.curr_indice + 1) % len(self.frame_list)]))

        self.addButton = QPushButton("+")
        self.addButton.setEnabled(True)
        self.addButton.setStatusTip('add a frame for correction')
        self.addButton.clicked.connect(self.add_frame)

        self.removeButton = QPushButton("-")
        self.removeButton.setEnabled(True)
        self.removeButton.setStatusTip('remove a frame from correction')
        self.removeButton.clicked.connect(self.remove_frame)

        self.copyButton = QPushButton("Copy")
        self.copyButton.setEnabled(True)
        self.copyButton.setStatusTip('Copy current attention')
        self.copyButton.clicked.connect(self.copy_frame)

        self.pasteButton = QPushButton("Paste")
        self.pasteButton.setEnabled(True)
        self.pasteButton.setStatusTip('Paste')
        self.pasteButton.clicked.connect(self.paste_frame)
        
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
        self.saveButton.setShortcut("Ctrl+S")
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

        ## Correction print
        self.showCorrectButton = QPushButton("&Show Corrected")
        self.showCorrectButton.setEnabled(True)
        self.showCorrectButton.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        self.showCorrectButton.clicked.connect(self.showCorrected)

        ## Finish button 
        self.finishButton = QPushButton("&Save and Finish")
        self.finishButton.setEnabled(True)
        self.finishButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
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
        layoutFrameList.addWidget(self.copyButton)
        layoutFrameList.addWidget(self.pasteButton)

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

        corrLayout = QHBoxLayout()
        corrLayout.addWidget(self.project3dButton)
        corrLayout.addWidget(self.showCorrectButton)

        subLayout = QVBoxLayout()
        subLayout.addLayout(featureLayout)
        subLayout.addLayout(corrLayout)
        subLayout.addWidget(self.finishButton)

        mainLayout = QHBoxLayout()     
        mainLayout.addLayout(layoutLeft)
        mainLayout.addLayout(inputLayout)
        mainLayout.addLayout(subLayout)

        self.setLayout(mainLayout)
        self.update_frame()

    def setHW(self, h, w):
        cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_zoom_out.npy')
        self.cams = read_cameras(cam_file)
        self.h, self.w = 2160, 1920
        return

    def setCams(self, cam_id):
        cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_zoom_out.npy')
        if cam_id == 1:
            cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_zoom_in.npy')
        self.cams = read_cameras(cam_file)
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
        """if value not in self.viewer3D.attention:
            print("Frame does not have attention")
            return  """          
        if ok:
            value = int(value)
            self.frame_list = np.append(self.frame_list, value)
            self.frame_list = np.sort(self.frame_list)
            print(self.frame_list)
            
            self.frame_listW.addItem(ListWidgetItem("{} - NA".format(value)))
            self.curr_indice = self.frame_listW.currentRow()
            if len(self.frame_list) == 1: 
                self.frame_listW.setCurrentRow(0)
                self.curr_indice = 0
                self.update_frame()

    def remove_frame(self):
        if len(self.frame_list) == 0: 
            return
        f = self.frame_list[self.curr_indice]
        self.frame_list = np.delete(self.frame_list,self.curr_indice)
        self.frame_listW.takeItem(self.curr_indice)
        self.curr_indice = self.frame_listW.currentRow()
        if f in self.corrected_list:
            self.corrected_list.remove(f)
        if len(self.frame_list) == 0: 
            self.curr_indice = -1
            self.framelabel.setText("No Frames")
        self.update_frame()

    def copy_frame(self):
        if len(self.frame_list) == 0 or self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        if not curr_frame in self.viewer3D.attention:
            return
        data = self.viewer3D.attention[curr_frame]
        self.memory_buffer = copy.deepcopy(data)
        return

    def paste_frame(self):
        if self.memory_buffer is None: return
        curr_frame = self.frame_list[self.curr_indice]
        if not curr_frame in self.viewer3D.attention:
            return        

        self.viewer3D.attention[curr_frame] = copy.deepcopy(self.memory_buffer)
        self.update_frame() 
        self.save_pos()
        return
    
    def update_info(self):
        if len(self.frame_list) == 0: return
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

        #self.origin_vec = (data["u"][1] - data["u"][0])/np.linalg.norm(data["u"][1] - data["u"][0])
        #self.change_att_direction(self.origin_vec)

        return

    def change_attention_sig(self, att):
        self.old_x_att, self.old_y_att, self.old_z_att = att[0], att[1], att[2]
        self.max_XAEdit.setValue(att[0])
        self.max_YAEdit.setValue(att[1])
        self.max_ZAEdit.setValue(att[2])
        return

    def change_position_sig(self, pos):
        self.old_x, self.old_y, self.old_z = pos[0], pos[1], pos[2]
        self.max_XEdit.setValue(pos[0])
        self.max_YEdit.setValue(pos[1])
        self.max_ZEdit.setValue(pos[2])
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
        new_vec = self.viewer3D.translate_head(value - self.old_x, 0.0, 0.0)
        self.old_x = value
        self.modify_att = False
        self.change_att_direction(new_vec)
        self.modify_att = True
        return

    def y_changed(self, box, value):
        new_vec = self.viewer3D.translate_head(0.0, value - self.old_y, 0.0)
        self.old_y = value
        self.modify_att = False
        self.change_att_direction(new_vec)  
        self.modify_att = True     
        return

    def z_changed(self, box, value):
        new_vec = self.viewer3D.translate_head(0.0, 0.0, value - self.old_z)
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
        new_vec = self.viewer3D.translate_attention_p(value - self.old_x_att, 0.0, 0.0)
        self.old_x_att = value
        self.change_att_direction(new_vec)
        return

    def y_att_changed(self, box, value):
        new_vec = self.viewer3D.translate_attention_p(0.0, value - self.old_y_att, 0.0)
        self.old_y_att = value
        self.change_att_direction(new_vec)
        return

    def z_att_changed(self, box, value):
        new_vec = self.viewer3D.translate_attention_p(0.0, 0.0, value - self.old_z_att)
        self.old_z_att = value
        self.change_att_direction(new_vec)
        return

    def yaw_changed(self, box, value):
        att = self.viewer3D.rotate_attention(value - self.old_yaw, "x", self.modify_att)
        self.old_yaw = value
        self.update_att(att)
        return

    def pitch_changed(self, box, value):
        att = self.viewer3D.rotate_attention(value - self.old_pitch, "y", self.modify_att)
        self.old_pitch = value
        self.update_att(att)
        return

    def roll_changed(self, box, value):
        att = self.viewer3D.rotate_attention(value - self.old_roll, "z", self.modify_att)
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
        #curr_frame = self.frame_list[self.curr_indice]
        self.frame_id.emit(curr_frame)
        if curr_frame not in self.corrected_list:
            self.corrected_list.add(curr_frame)
        self.frame_listW.item(self.curr_indice).setBackground(Qt.green)
        return

    def project3D(self, data):
        if len(data) < 2: return
        if self.curr_indice == -1: return
        item = self.viewer3D.current_item
        
        print(data)
    
        att = to_3D(data, self.cams, self.h, self.w)
        u = att - item["head"].pos[0]
        u = u / np.linalg.norm(u)
        att = self.viewer3D.collision(item["head"].pos[0], u)
 
        p = {"pos":item["head"].pos[0], "att":att}
        poses = project_2d(p, self.cams, self.h, self.w)

        new_vec = self.viewer3D.translate_attention_p(att[0] - self.old_x_att, att[1] - self.old_y_att, att[2] - self.old_z_att)
        self.old_x_att,  self.old_y_att,  self.old_z_att = att[0], att[1], att[2]
        self.change_att_direction(new_vec)

        self.pose2d.emit(poses)
        return

    def project2D(self):    
        if self.curr_indice == -1: return
        item = self.viewer3D.current_item
        pos = item["head"].pos[0]
        u = item["att"].pos[0] - pos
        u = u / np.linalg.norm(u)
        att = self.viewer3D.collision(pos, u)

        p = {"pos":pos, "att":att}
        poses = project_2d(p, self.cams, self.h, self.w)
        self.pose2d.emit(poses)
        return

    def propagate(self, threshold = 30):
        if len(self.corrected_list) == 0: 
            print("No frame corrected")
            return 

        self.corrected_list = sorted(self.corrected_list)
        f = self.corrected_list.pop()
        corrected_merged = [[f]]
        self.viewer3D.attention[f]["corrected_flag"] = 1
        for f in self.corrected_list:
            self.viewer3D.attention[f]["corrected_flag"] = 1
            prev = corrected_merged[-1][-1]
            if abs(f-prev) < threshold:
                corrected_merged[-1].append(f)
            else:
                corrected_merged.append([f])

        min_f, max_f = min(self.viewer3D.attention.keys()), max(self.viewer3D.attention.keys())
        for seg in corrected_merged: 
            seg = sorted(seg)        
            start, end = max(min_f, seg[0]-threshold), min(seg[-1] + threshold, max_f)
            print(seg, start, end)
            mask = build_mask([x-start for x in seg], end-start+1, threshold = threshold)[:, np.newaxis]
            interp_list = [start] + seg + [end]
            interp_poses = []
            for f in interp_list:
                p = self.viewer3D.attention[f]
                h, v = p["u"][0], p["u"][1]-p["u"][0]
                v_n = np.linalg.norm(v)
                if v_n <= 1e-6:
                    info = interp_poses[-1]
                else:
                    v = v/v_n
                    info = np.concatenate([h, v], axis=0)
                interp_poses.append(info) 
            interp_poses = np.array(interp_poses)   
            interp_func = interpolate.interp1d(interp_list, interp_poses, axis=0, kind = 'quadratic')
            x_interp = interp_func(np.arange(start, end, 1))
            old_p = None
            for i, f in enumerate(range(start, end)):
                p = self.viewer3D.attention[f]
                m = mask[i]
                v_or = np.copy(p["u"][1] - p["u"][0])
                p["head"] = (1-m)*p["head"] + x_interp[i, :3]*m
                v = (1-m)*v_or + x_interp[i, 3:]*m
                v_n = np.linalg.norm(v)
                if v_n <= 1e-6:
                    v = np.copy(p["u"][1] - p["u"][0])
                    v = v/np.linalg.norm(v)
                else:
                    v = v/v_n          
                att = self.viewer3D.collision(p["head"], v)
                if (att is None or v is None or p["head"] is None) and old_p is not None:
                    p = copy.deepcopy(old_p)
                    continue 
                p["u"][0], p["line"][0] = p["head"], p["head"]
                p["u"][1] = p["head"] + v*5.0
                p["line"][1] = att
                p["att"] = att
                size = np.linalg.norm(p["head"] - att)
                p["size"] = np.clip(size*4.0, 10.0, 80.0)
                old_p = p                  
        return 

    def runGP(self):
        x_tr, frame_list = [], []
        for i, (f, p) in enumerate(self.viewer3D.attention.items()):
            h, v = p["u"][0], p["u"][1]-p["u"][0]
            v_n = np.linalg.norm(v)
            if v_n <= 1e-6:
                info = x_tr[-1]
            else:
                v = v/v_n
                info = np.concatenate([h, v], axis=0)
            x_tr.append(info)
            frame_list.append(f)

        self.write_attention("temp.txt")
        N = 60 #len(self.viewer3D.attention) // 1800
        uncertain_frames, uncertain_scores = get_uncertainty(x_tr, max_n= N * 2)
        uncertain_frames = np.array([frame_list[f] for f in uncertain_frames])
        ind = uncertain_frames.argsort()
        uncertain_scores = uncertain_scores[ind]
        self.frame_list = uncertain_frames[ind]
        print(self.frame_list)
        print("{} Frames proposed to correct, around 2 frames/min to correct".format(len(self.frame_list)))
        if len(self.frame_list) == 0:
            self.curr_indice = -1
            self.frame_listW.clear()
            self.corrected_list = set()
            return

        self.curr_indice = 0
        self.frame_listW.clear()
        for f, s in zip(self.frame_list, uncertain_scores):
            self.frame_listW.addItem(ListWidgetItem("{} - {:.2f}".format(f,s)))
        self.frame_listW.setCurrentRow(0)
        self.corrected_list = set()
        self.update_frame()
        return

    def showCorrected(self):
        if len(self.history_corrected) == 0:
            message = "No frames"
        else:
            message = ", ".join([str(x) for x in sorted(self.history_corrected)])
        print(message)
        QMessageBox.about(self, "List of corrected frames", message)
        return

    def write_attention(self, fileName = None, new_att = []):
        if fileName is None:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Corrected Results", QDir.homePath() + "/corrected.txt", "Text files (*.txt)")
            #options=QFileDialog.DontUseNativeDialog)
            if fileName == '':
                return
        with open(fileName, "w") as w:
            w.write("")
            for i, (f, p) in enumerate(self.viewer3D.attention.items()):
                pos, v = p["u"][0], p["u"][1]-p["u"][0]
                att = p["att"]
                flag = p["corrected_flag"]
                w.write("{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:d}\n".format(
                    f, pos[0], pos[1], pos[2], v[0], v[1], v[2], att[0], att[1], att[2], flag
                ))
                if flag: self.history_corrected.add(f)
        self.viewer3D.read_attention(fileName)
        print("Corrected frames:", len(self.history_corrected))
        print("File saved")
        
        return

    def finish(self):
        self.propagate()
        self.write_attention()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you want to Run the assistant?")
        msg.setWindowTitle("Assistant")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
        retval = msg.exec_()
        if retval == QMessageBox.Yes:
            self.runGP()      
        return

    def closeEvent(self, event):
        self.pose2d.emit({})
        self.open_id.emit(False)
        event.accept()