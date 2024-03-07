from PyQt5.QtWidgets import QStyle, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QDialog, QMessageBox, QListWidget, \
                                QAbstractItemView, QInputDialog, QListWidgetItem, QCheckBox, QComboBox
from PyQt5.QtCore import pyqtSignal, QDir, Qt
#from PyQt5.QtGui import QMessageBox
import pyqtgraph as pg
import copy
import pkg_resources
import numpy as np
from scipy import interpolate

from .GaussianProcess import get_uncertainty
from .utils import project_2d, build_mask, to_3D, write_results
from .ThreeIntWidget import ThreeEntryDialog

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

class CorrectionWindowHand(QWidget):
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

        self.setWindowTitle("Correction Tool Hands") 
        self.resize(400 , 200 )

        self.viewer3D = viewer3D
        self.frame_list = np.array([], dtype=int)
        self.curr_indice = -1
        self.old_x_r, self.old_y_r, self.old_z_r = 0, 0, 0
        self.old_x_l, self.old_y_l, self.old_z_l = 0, 0, 0
        self.corrected_list = set()
        self.history_corrected = self.viewer3D.corrected_frames_hand
        self.modify_att = True
        self.memory_buffer = None
        self.segmentIndex = 0
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

        self.addRangeButton = QPushButton("[++]")
        self.addRangeButton.setEnabled(True)
        self.addRangeButton.setStatusTip('Add multiple frames for correction at a fixed rate within a range')
        self.addRangeButton.clicked.connect(self.add_frame_range)
        
        self.addManyButton = QPushButton("++")
        self.addManyButton.setEnabled(True)
        self.addManyButton.setStatusTip('Add multiple frames for correction at a fixed rate')
        self.addManyButton.clicked.connect(self.add_frame_many)

        self.addNeighborButton = QPushButton("[+]")
        self.addNeighborButton.setEnabled(True)
        self.addNeighborButton.setStatusTip('Add neighbor frames for correction')
        self.addNeighborButton.clicked.connect(self.add_neigh_frame)

        self.addCurrentButton = QPushButton("+Current")
        self.addCurrentButton.setEnabled(True)
        self.addCurrentButton.setStatusTip('Add currrent frame for correction')
        self.addCurrentButton.clicked.connect(self.add_current_frame)
               
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
        self.copyButton.setStatusTip('Copy current hands')
        self.copyButton.clicked.connect(self.copy_frame)

        self.pasteButton = QPushButton("Paste")
        self.pasteButton.setEnabled(True)
        self.pasteButton.setStatusTip('Paste copied hands')
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
        self.runGPButton = QPushButton("&Run Assistant - 60")
        self.runGPButton.setEnabled(True)
        self.runGPButton.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.runGPButton.clicked.connect(lambda checked, param=0: self.runGP(param))
        
        ## Run Assistant button 
        self.runGPButton2 = QPushButton("&Run Assistant - All")
        self.runGPButton2.setEnabled(True)
        self.runGPButton2.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.runGPButton2.clicked.connect(lambda checked, param=1: self.runGP(param))

        ## Project 3d button Left Hand
        self.project3dButtonLeft = QPushButton("&Project 3D: Left")
        self.project3dButtonLeft.setEnabled(True)
        self.project3dButtonLeft.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        #color_tuple = (0.9,0.5,0.2)  # RGB tuple (values between 0.0 and 1.0)
        color_tuple = (1.0,0.0,1.0)
        self.project3dButtonLeft.setStyleSheet(f'QPushButton {{background-color: rgb({int(color_tuple[0]*255)}, \
            {int(color_tuple[1]*255)}, {int(color_tuple[2]*255)}); color:black;}}')


        ## Project 3d button Right Hand
        self.project3dButtonRight = QPushButton("&Project 3D: Right")
        self.project3dButtonRight.setEnabled(True)
        self.project3dButtonRight.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        #color_tuple = (0.9, 1.0, 0.0)  # RGB tuple (values between 0.0 and 1.0)
        color_tuple = (0.0,1.0,0.0)
        self.project3dButtonRight.setStyleSheet(f'QPushButton {{background-color: rgb({int(color_tuple[0]*255)}, \
            {int(color_tuple[1]*255)}, {int(color_tuple[2]*255)}); color: black;}}')

        ## Correction print
        self.showCorrectButton = QPushButton("&Info")
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
        self.max_XREdit = pg.SpinBox()
        self.max_XREdit.setDecimals(3)
        self.max_XREdit.setRange(-15.0, 15.0)
        self.max_XREdit.setMinimumSize(80,30)
        self.max_XREdit.setSingleStep(0.05)
        self.max_XREdit.sigValueChanging.connect(self.x_r_changed)
        #self.max_XEdit.editingFinished.connect(self.save_pos)

        self.max_YREdit = pg.SpinBox()
        self.max_YREdit.setDecimals(3)
        self.max_YREdit.setRange(-20.0, 20.0)
        self.max_YREdit.setMinimumSize(80,30)
        self.max_YREdit.setSingleStep(0.05)
        self.max_YREdit.sigValueChanging.connect(self.y_r_changed)
        #self.max_YEdit.editingFinished.connect(self.save_pos)

        self.max_ZREdit = pg.SpinBox()
        self.max_ZREdit.setDecimals(3)
        self.max_ZREdit.setRange(0.0, 10.0)
        self.max_ZREdit.setMinimumSize(80,30)
        self.max_ZREdit.setSingleStep(0.05)
        self.max_ZREdit.sigValueChanging.connect(self.z_r_changed)
        #self.max_ZEdit.editingFinished.connect(self.save_pos)

        ## Input editing
        self.max_XLEdit = pg.SpinBox()
        self.max_XLEdit.setDecimals(3)
        self.max_XLEdit.setRange(-15.0, 15.0)
        self.max_XLEdit.setMinimumSize(80,30)
        self.max_XLEdit.setSingleStep(0.01)
        self.max_XLEdit.sigValueChanging.connect(self.x_l_changed)
        #self.max_XEdit.editingFinished.connect(self.save_pos)

        self.max_YLEdit = pg.SpinBox()
        self.max_YLEdit.setDecimals(3)
        self.max_YLEdit.setRange(-20.0, 20.0)
        self.max_YLEdit.setMinimumSize(80,30)
        self.max_YLEdit.setSingleStep(0.01)
        self.max_YLEdit.sigValueChanging.connect(self.y_l_changed)
        #self.max_YEdit.editingFinished.connect(self.save_pos)

        self.max_ZLEdit = pg.SpinBox()
        self.max_ZLEdit.setDecimals(3)
        self.max_ZLEdit.setRange(0.0, 20.0)
        self.max_ZLEdit.setMinimumSize(80,30)
        self.max_ZLEdit.setSingleStep(0.01)
        self.max_ZLEdit.sigValueChanging.connect(self.z_l_changed)
        #self.max_ZEdit.editingFinished.connect(self.save_pos)

        # Label
        self.xllabel = QLabel("X Left")
        self.xllabel.setBuddy(self.max_XLEdit)
        self.yllabel = QLabel("Y Left")
        self.yllabel.setBuddy(self.max_YLEdit)
        self.zllabel = QLabel("Z Left")
        self.zllabel.setBuddy(self.max_ZLEdit)
        
        self.xrlabel = QLabel("X Right")
        self.xrlabel.setBuddy(self.max_XREdit)
        self.yrlabel = QLabel("Y Right")
        self.yrlabel.setBuddy(self.max_YREdit)
        self.zrlabel = QLabel("Z Right")
        self.zrlabel.setBuddy(self.max_ZREdit)

        # Segment Combo bos
        self.ComboBox = QComboBox()
        self.ComboBox.addItem('All Segments')
        self.ComboBox.currentIndexChanged.connect(self.index_changed_combo)
        
        # Layout
        layoutButton = QHBoxLayout()
        layoutButton.addWidget(self.prevframeButton)
        layoutButton.addWidget(self.frameButton)
        layoutButton.addWidget(self.nextframeButton)


        layoutFrameList = QVBoxLayout()
        layoutFrameList.addWidget(self.framelabel)
        layoutFrameList.addWidget(self.addRangeButton)
        layoutFrameList.addWidget(self.addManyButton)
        layoutFrameList.addWidget(self.addNeighborButton)
        layoutFrameList.addWidget(self.addCurrentButton)
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

        layoutPosL = QHBoxLayout()
        layoutPosL.addWidget(self.xllabel)
        layoutPosL.addWidget(self.max_XLEdit)
        layoutPosL.addWidget(self.yllabel)
        layoutPosL.addWidget(self.max_YLEdit)
        layoutPosL.addWidget(self.zllabel)
        layoutPosL.addWidget(self.max_ZLEdit)
        
        layoutPosR = QHBoxLayout()
        layoutPosR.addWidget(self.xrlabel)
        layoutPosR.addWidget(self.max_XREdit)
        layoutPosR.addWidget(self.yrlabel)
        layoutPosR.addWidget(self.max_YREdit)
        layoutPosR.addWidget(self.zrlabel)
        layoutPosR.addWidget(self.max_ZREdit)

        inputLayout = QVBoxLayout()
        inputLayout.addLayout(layoutPosL)
        inputLayout.addLayout(layoutPosR)
        
        featureLayout = QHBoxLayout()
        featureLayout.addWidget(self.project2dButton)
        featureLayout.addWidget(self.runGPButton)
        featureLayout.addWidget(self.runGPButton2)

        corrLayout = QHBoxLayout()
        corrLayout.addWidget(self.project3dButtonLeft)
        corrLayout.addWidget(self.project3dButtonRight)
        corrLayout.addWidget(self.showCorrectButton)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.ComboBox)

        subLayout = QVBoxLayout()
        subLayout.addLayout(topLayout)
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
        self.setCams(0)
        self.h, self.w = 2160, 1920
        return

    def setCams(self, cam_id):
        cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_zoom_out.npy')
        if cam_id == 1:
            cam_file = pkg_resources.resource_filename('DevEv', 'metadata/CameraParameters/camera_zoom_in.npy')
        self.cams = read_cameras(cam_file)
        self.cam_id = cam_id
        return

    def index_changed_combo(self, index):
        self.segmentIndex = index
        return
    
    def update_combobox(self):
        self.ComboBox.clear()
        self.ComboBox.addItem('All Segments')        
        segments = self.viewer3D.segment
        if segments is None: return
        for i, (s,e) in enumerate(segments):
            self.ComboBox.addItem('S{} -> {} -{}'.format(i, s, e))
        self.ComboBox.setCurrentIndex(self.segmentIndex) 
        return
        

    def select_frame(self, item):
        self.curr_indice = self.frame_listW.currentRow()
        self.update_frame()
        return

    def add_frame_range(self):
        dialog = ThreeEntryDialog(self)
        ok = dialog.exec_() 
        
        if ok == QDialog.Accepted: 
            start, end, step = dialog.getInputs()
            L = list(self.viewer3D.attention.keys())
            for x in np.arange(start, end, step, dtype=int):
                if x in self.frame_list or x not in L: continue
                
                self.frame_list = np.append(self.frame_list, x)
                self.frame_listW.addItem(ListWidgetItem("{} - NA".format(x)))
            self.frame_list = np.sort(self.frame_list)
            self.frame_listW.setCurrentRow(0)
            self.curr_indice = 0
            self.update_frame()
            
    def add_current_frame(self):
        frame = self.viewer3D.current_item["frame"]     
        self._include_frame(frame)
        return

    def add_neigh_frame(self):
        value, ok = QInputDialog.getInt(self, 'Add past and future of selected frame', 'Enter an offset \n(single entry)')
        if not ok: return
        frame = self.viewer3D.current_item["frame"]       
        self._include_frame(frame - int(value))
        self._include_frame(frame + int(value))
        return        
    
    def _include_frame(self, frame):
        if frame in self.frame_list:
            print("Frame already selected")
            return
        if frame not in self.viewer3D.attention:
            print("Frame does not have attention")
            return  
        self.frame_list = np.append(self.frame_list, frame)
        self.frame_list = np.sort(self.frame_list)
        print(self.frame_list)
        
        self.frame_listW.addItem(ListWidgetItem("{} - NA".format(frame)))
        self.curr_indice = self.frame_listW.currentRow()
        if len(self.frame_list) == 1: 
            self.frame_listW.setCurrentRow(0)
            self.curr_indice = 0
            self.update_frame()
        return
                       
    def add_frame(self):
        value, ok = QInputDialog.getInt(self, 'Add frame', 'Enter a frame number to add \n(single entry)')
        if ok:
            self._include_frame(int(value))
            
    def add_frame_many(self):
        value, ok = QInputDialog.getInt(self, 'Add frames at fixed rate', 'Enter a frame rate \n(single entry)')
        L = list(self.viewer3D.attention.keys())
        if value <= 5:
            print("Frame rate too small")
            return
        if value >= max(L):
            print("Frame rate too high")
            return          
        if ok:
            if self.segmentIndex == 0: 
                start, end = min(L), max(L)
            else:
                start, end = self.viewer3D.segment[self.segmentIndex-1]       
            value = int(value)
            for x in np.arange(start, end, value, dtype=int):
                if not x in L or x in self.frame_list: continue
                self.frame_list = np.append(self.frame_list, x)
                self.frame_listW.addItem(ListWidgetItem("{} - NA".format(x)))
            self.frame_list = np.sort(self.frame_list)
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

        pos_HL = data["handL"]
        self.old_x_l, self.old_y_l, self.old_z_l = pos_HL[0], pos_HL[1], pos_HL[2]
        self.max_XLEdit.setValue(pos_HL[0])
        self.max_YLEdit.setValue(pos_HL[1])
        self.max_ZLEdit.setValue(pos_HL[2])
        
        pos_HR = data["handR"]
        self.old_x_r, self.old_y_r, self.old_z_r = pos_HR[0], pos_HR[1], pos_HR[2]
        self.max_XREdit.setValue(pos_HR[0])
        self.max_YREdit.setValue(pos_HR[1])
        self.max_ZREdit.setValue(pos_HR[2])
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

    def x_r_changed(self, box, value):
        self.viewer3D.translate_hand_right(value - self.old_x_r, 0.0, 0.0)
        self.old_x_r = value
        return

    def y_r_changed(self, box, value):
        self.viewer3D.translate_hand_right(0.0, value - self.old_y_r, 0.0)
        self.old_y_r = value
        return

    def z_r_changed(self, box, value):
        self.viewer3D.translate_hand_right(0.0, 0.0, value - self.old_z_r)
        self.old_z_r = value
        return

    def x_l_changed(self, box, value):
        self.viewer3D.translate_hand_left(value - self.old_x_l, 0.0, 0.0)
        self.old_x_l = value
        return

    def y_l_changed(self, box, value):
        self.viewer3D.translate_hand_left(0.0, value - self.old_y_l, 0.0)
        self.old_y_l = value
        return

    def z_l_changed(self, box, value):
        self.viewer3D.translate_hand_left(0.0, 0.0, value - self.old_z_l)
        self.old_z_l = value
        return

    def save_pos(self):
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        handL_changed, handR_changed = self.viewer3D.save_hands(curr_frame)
        
        self.frame_id.emit(curr_frame)
        if curr_frame not in self.corrected_list:
            self.corrected_list.add(curr_frame)
        self.frame_listW.item(self.curr_indice).setBackground(Qt.green)
        self.project2D(update_2d = True)
        return

    def project3D(self, data):
        if len(data) < 3: return
        if self.curr_indice == -1: return
        item = self.viewer3D.current_item
        
        att = to_3D(data, self.cams, self.h, self.w)
        p = {}
        if data["type"] == "handL":
            self.viewer3D.translate_hand_left(att[0] - self.old_x_l, att[1] - self.old_y_l, att[2] - self.old_z_l)
            self.old_x_l,  self.old_y_l,  self.old_z_l = att[0], att[1], att[2]
            p = {"handL":att, "handR":item["hand"].pos[1]} 
        elif data["type"] == "handR":    
            self.viewer3D.translate_hand_right(att[0] - self.old_x_r, att[1] - self.old_y_r, att[2] - self.old_z_r)
            self.old_x_r,  self.old_y_r,  self.old_z_r = att[0], att[1], att[2]
            p = {"handL":item["hand"].pos[0], "handR":att}
        else: return       
        poses = project_2d(p, self.cams, self.h, self.w, is_mat = self.cam_id == 1)
        self.pose2d.emit(poses)
        return

    def project2D(self, update_2d = False):    
        if self.curr_indice == -1: return
        item = self.viewer3D.current_item

        p = {"handL":item["hand"].pos[0], "handR":item["hand"].pos[1]}
        poses = project_2d(p, self.cams, self.h, self.w, is_mat = self.cam_id == 1)
        poses["update"] = update_2d
        self.pose2d.emit(poses)
        return 

    def propagate(self, threshold = 30):
        if len(self.corrected_list) == 0: 
            print("No frame corrected")
            return 

        corrected_list = sorted(self.corrected_list)
        f = corrected_list[0]
        corrected_merged = [[f]]

        self.viewer3D.attention[f]["corrected_flag_hand"] = 1
        for f in corrected_list[1:]:
            self.viewer3D.attention[f]["corrected_flag_hand"] = 1
            prev = corrected_merged[-1][-1]
            if abs(f-prev) < threshold:
                corrected_merged[-1].append(f)
            else:
                corrected_merged.append([f])
        print(corrected_merged)        
        self.write_attention("temp.txt", is_temp=True)
        
        min_f, max_f = min(self.viewer3D.attention.keys()), max(self.viewer3D.attention.keys())
        for seg in corrected_merged: 
            seg = sorted(seg)        
            start, end = max(min_f, seg[0]-threshold), min(seg[-1] + threshold, max_f)
            while start not in self.viewer3D.attention:
                start += 1
            while end not in self.viewer3D.attention:
                end -= 1
                                
            print(seg, start, end)
            mask = build_mask([x-start for x in seg], end-start+1, threshold = threshold)[:, np.newaxis]
            interp_list = np.unique([start] + seg + [end])
            interp_poses = []
            for f in interp_list:
                p = self.viewer3D.attention[f]
                h, v = p["handL"], p["handR"]
                info = np.concatenate([h, v], axis=0)
                interp_poses.append(info) 
            interp_poses = np.array(interp_poses)   
            if len(interp_list) == 2:
                interp_func = interpolate.interp1d(interp_list, interp_poses, axis=0, kind = 'linear')
            else: interp_func = interpolate.interp1d(interp_list, interp_poses, axis=0, kind = 'quadratic')
            x_interp = interp_func(np.arange(start, end, 1))
            old_p = None
            
            for i, f in enumerate(range(start, end)):
                p = self.viewer3D.attention[f]
                m = mask[i]

                p["handL"] = (1-m)*p["handL"] + x_interp[i, :3]*m
                p["handR"] = (1-m)*p["handR"] + x_interp[i, 3:]*m

                if p["handL"] is None and old_p is not None:
                    p = copy.deepcopy(old_p)
                    continue 
                old_p = p                  
        return 

    def runGP(self, param):
        x_tr, frame_list = [], []
        for i, (f, p) in enumerate(self.viewer3D.attention.items()):
            h, v = p["handL"], p["handR"]
            info = np.concatenate([h, v], axis=0)
            if self.segmentIndex == 0: 
                frame_list.append(f)
                x_tr.append(info)
            else:
                start, end = self.viewer3D.segment[self.segmentIndex-1]
                if start <= f <= end: 
                    frame_list.append(f)
                    x_tr.append(info)

        self.write_attention("temp.txt", is_temp=True)
        N = 60 if param == 0 else len(self.viewer3D.attention) // 1800
        uncertain_frames, uncertain_scores = get_uncertainty(x_tr, max_n= N)
        uncertain_frames = np.array([frame_list[f] for f in uncertain_frames])
        ind = uncertain_frames.argsort()
        uncertain_scores = uncertain_scores[ind]
        self.frame_list = uncertain_frames[ind]
        print(self.frame_list)
        print("{} Frames proposed to correct, for finetuning".format(len(self.frame_list)))
        if len(self.frame_list) == 0:
            self.curr_indice = -1
            self.frame_listW.clear()
            self.corrected_list = set()
            return

        self.curr_indice = 0
        self.frame_listW.clear()
        for f, s in zip(self.frame_list, uncertain_scores):
            self.frame_listW.addItem(ListWidgetItem("{} - {:.2f}".format(f,s)))
            self.viewer3D.attention[f]["corrected_flag_hand"] = 2
        self.frame_listW.setCurrentRow(0)
        self.corrected_list = set()
        self.update_frame()
        return
    
    def showCorrected(self):
        message = ''
        if self.viewer3D.segment is None:
            message += "No segments\n"
        else:
            for i, (s,e) in enumerate(self.viewer3D.segment):
                message += "Segment {} -> {} - {}\n".format(i, s, e)
        if len(self.history_corrected) == 0:
            message += "\nNo Corrected Frames"
        else:
            message += '\n{} Corrected Frames:\n'.format(len(self.history_corrected))
            message += ", ".join([ str(x) for x, y in self.history_corrected.items() if y == 1])
        print(message)
        QMessageBox.about(self, "Info Hands", message)
        return

    def write_attention(self, fileName = None, is_temp = False):
        write_results(self, "hand", fileName = fileName, is_temp = is_temp)
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
            self.runGP(0)      
        return

    def update_list_frames(self):
        self.history_corrected = self.viewer3D.corrected_frames_hand
        print("History of Corrected frames for hands", self.history_corrected)
        self.frame_listW.clear()
        self.frame_list = np.array([], dtype=int)
        
        for value, flag in self.history_corrected.items():
            value = int(value)
            self.frame_list = np.append(self.frame_list, value)
            self.frame_listW.addItem(ListWidgetItem("{} - NA".format(value)))
            if flag == 1:
                self.frame_listW.item(len(self.frame_list) - 1).setBackground(Qt.blue)
            
        self.frame_list = np.sort(self.frame_list)
        if len(self.history_corrected) > 0: 
            self.frame_listW.setCurrentRow(0)
            self.curr_indice = 0
        else:
            self.curr_indice = -1
        self.update_frame()
        self.update_combobox()
        return

    def closeEvent(self, event):
        self.pose2d.emit({})
        self.open_id.emit(False)
        event.accept()