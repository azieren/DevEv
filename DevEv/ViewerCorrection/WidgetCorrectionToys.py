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
from .utils import project_2d_simple, build_mask, to_3D, write_results_toy
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

class CorrectionWindowToys(QWidget):
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

        self.setWindowTitle("Correction Tool Toys") 
        self.resize(400 , 200 )

        self.viewer3D = viewer3D
        self.frame_list = np.array([], dtype=int)
        self.curr_indice = -1
        self.corrected_list = []
        self.history_corrected = self.viewer3D.corrected_frames_hand
        self.memory_buffer = None
        self.segmentIndex = 0
        ## Init cameras
        self.setHW(0, 0)

        self.frame_listW = QListWidget()
        self.frame_listW.setSelectionMode(QAbstractItemView.SingleSelection)
        for f in self.frame_list:
            self.frame_listW.addItem(ListWidgetItem("{:d} - NA".format(f)))
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
        self.project3dButton = QPushButton("&Project 3D")
        self.project3dButton.setEnabled(True)
        self.project3dButton.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        #color_tuple = (0.9,0.5,0.2)  # RGB tuple (values between 0.0 and 1.0)
        color_tuple = (1.0,1.0,0.0)
        self.project3dButton.setStyleSheet(f'QPushButton {{background-color: rgb({int(color_tuple[0]*255)}, \
            {int(color_tuple[1]*255)}, {int(color_tuple[2]*255)}); color:black;}}')

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
        self.max_XEdit = pg.SpinBox()
        self.max_XEdit.setDecimals(3)
        self.max_XEdit.setRange(-15.0, 15.0)
        self.max_XEdit.setMinimumSize(80,30)
        self.max_XEdit.setSingleStep(0.05)
        self.max_XEdit.sigValueChanging.connect(self.x_changed)
        #self.max_XEdit.editingFinished.connect(self.save_pos)

        self.max_YEdit = pg.SpinBox()
        self.max_YEdit.setDecimals(3)
        self.max_YEdit.setRange(-20.0, 20.0)
        self.max_YEdit.setMinimumSize(80,30)
        self.max_YEdit.setSingleStep(0.05)
        self.max_YEdit.sigValueChanging.connect(self.y_changed)
        #self.max_YEdit.editingFinished.connect(self.save_pos)

        self.max_ZEdit = pg.SpinBox()
        self.max_ZEdit.setDecimals(3)
        self.max_ZEdit.setRange(0.0, 10.0)
        self.max_ZEdit.setMinimumSize(80,30)
        self.max_ZEdit.setSingleStep(0.05)
        self.max_ZEdit.sigValueChanging.connect(self.z_changed)
        #self.max_ZEdit.editingFinished.connect(self.save_pos)


        # Label
        self.xlabel = QLabel("X")
        self.xlabel.setBuddy(self.max_XEdit)
        self.ylabel = QLabel("Y")
        self.ylabel.setBuddy(self.max_YEdit)
        self.zlabel = QLabel("Z")
        self.zlabel.setBuddy(self.max_ZEdit)

        # Segment Combo bos
        self.ComboBox = QComboBox()
        self.ComboBox.addItem('All Segments')
        self.ComboBox.currentIndexChanged.connect(self.index_changed_combo)

        # Segment Combo bos
        self.ToyBox = QComboBox()
        self.toy_list = []
        self.current_toy = 0
        for i, (name, obj) in enumerate(self.viewer3D.room.toy_objects.items()):
            self.corrected_list.append(set())
            self.ToyBox.addItem(name)
            self.toy_list.append({"name":name, "obj":obj})
        self.ToyBox.currentIndexChanged.connect(self.index_changed_toy)
        
               
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

        layoutPos = QHBoxLayout()
        layoutPos.addWidget(self.xlabel)
        layoutPos.addWidget(self.max_XEdit)
        layoutPos.addWidget(self.ylabel)
        layoutPos.addWidget(self.max_YEdit)
        layoutPos.addWidget(self.zlabel)
        layoutPos.addWidget(self.max_ZEdit)
        
        featureLayout = QHBoxLayout()
        featureLayout.addWidget(self.project2dButton)
        featureLayout.addWidget(self.runGPButton)
        featureLayout.addWidget(self.runGPButton2)

        corrLayout = QHBoxLayout()
        corrLayout.addWidget(self.project3dButton)
        corrLayout.addWidget(self.showCorrectButton)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.ComboBox)
        topLayout.addWidget(self.ToyBox)

        subLayout = QVBoxLayout()
        subLayout.addLayout(topLayout)
        subLayout.addLayout(layoutPos)
        subLayout.addLayout(featureLayout)
        subLayout.addLayout(corrLayout)
        subLayout.addWidget(self.finishButton)

        mainLayout = QHBoxLayout()     
        mainLayout.addLayout(layoutLeft)
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

    def index_changed_toy(self, index):
        name = self.toy_list[self.current_toy]["name"]
        self.viewer3D.room.toy_objects[name]["item"].opts['drawEdges'] = False
        self.current_toy = index
        self.update_info()
        name = self.toy_list[self.current_toy]["name"]
        self.viewer3D.room.toy_objects[name]["item"].opts['drawEdges'] = True
        
        self.frame_list = []
        self.frame_listW.clear()
        for i, f in enumerate(self.corrected_list[self.current_toy]):
            self.frame_listW.addItem(ListWidgetItem("{:d} - NA".format(f)))
            self.frame_listW.item(i).setBackground(Qt.green)
            self.frame_list.append(f)
        self.frame_listW.setCurrentRow(0)
        self.frame_list = np.sort(self.frame_list).astype(int)
        return
    
    def index_changed_combo(self, index):
        self.segmentIndex = index
        return
    
    def update_combo_toy(self):
        for i, toy in enumerate(self.toy_list):
            if len(toy["obj"]["data"]) > 0:
                self.ToyBox.model().item(i).setBackground(Qt.green)
        return
    
    def update_combobox(self):
        self.ComboBox.clear()
        self.ComboBox.addItem('All Segments')        
        segments = self.viewer3D.segment.current
        if segments is None: return
        for i, (stype, s,e) in enumerate(segments):
            self.ComboBox.addItem('S{} -> {} -{} ({})'.format(i, s, e, stype))
        self.ComboBox.setCurrentIndex(self.segmentIndex) 
        return
        

    def select_frame(self, item):
        self.curr_indice = self.frame_listW.currentRow()
        self.update_frame()
        return

    def add_frame_range(self):
        curr_toy = self.toy_list[self.current_toy]["obj"]

        dialog = ThreeEntryDialog(self)
        ok = dialog.exec_() 

        if ok == QDialog.Accepted: 
            start, end, step = dialog.getInputs()
            L = list(curr_toy["data"].keys())
            for x in np.arange(start, end, step, dtype=int):
                if x in self.frame_list: continue
                if x not in curr_toy["data"]:
                    curr_toy["data"][x] = {}
                self.frame_list = np.append(self.frame_list, x)
                self.frame_listW.addItem(ListWidgetItem("{:d} - NA".format(x)))
            self.frame_list = np.sort(self.frame_list).astype(int)
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
        curr_toy = self.toy_list[self.current_toy]["obj"]
        if frame not in curr_toy["data"]:
            curr_toy["data"][frame] = {}
        self.frame_list = np.append(self.frame_list, frame)
        self.frame_list = np.sort(self.frame_list).astype(int)
  
        self.frame_listW.addItem(ListWidgetItem("{:d} - NA".format(frame)))
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
        curr_toy = self.toy_list[self.current_toy]["obj"]
        if len(curr_toy["data"]) == 0: return
        L = list(curr_toy["data"].keys())
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
                _, start, end = self.viewer3D.segment.current[self.segmentIndex-1]       
            value = int(value)
            for x in np.arange(start, end, value, dtype=int):
                if not x in L or x in self.frame_list: continue
                self.frame_list = np.append(self.frame_list, x)
                self.frame_listW.addItem(ListWidgetItem("{:d} - NA".format(x)))
            self.frame_list = np.sort(self.frame_list).astype(int)
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
        if f in self.corrected_list[self.current_toy]:
            self.corrected_list[self.current_toy].remove(f)
        if len(self.frame_list) == 0: 
            self.curr_indice = -1
            self.framelabel.setText("No Frames")
        self.update_frame()

    def copy_frame(self):
        if len(self.frame_list) == 0 or self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        curr_toy = self.toy_list[self.current_toy]["obj"]
        if not curr_frame in curr_toy["data"] and "p3d" not in curr_toy["data"][curr_frame]:
            return
        data = curr_toy["data"][curr_frame]
        self.memory_buffer = copy.deepcopy(data)
        return

    def paste_frame(self):
        if self.memory_buffer is None: return
        curr_frame = self.frame_list[self.curr_indice]
        curr_toy = self.toy_list[self.current_toy]["obj"]

        curr_toy["data"][curr_frame] = copy.deepcopy(self.memory_buffer)
        self.update_frame() 
        self.save_pos()
        return
    
    def update_info(self):
        if len(self.frame_list) == 0: return
        curr_frame = self.frame_list[self.curr_indice]
        curr_toy = self.toy_list[self.current_toy]["obj"]
        if not curr_frame in curr_toy["data"]:
            return
        x, y, z = self.toy_list[self.current_toy]["obj"]["center"]
        self.max_XEdit.setValue(x)
        self.max_YEdit.setValue(y)
        self.max_ZEdit.setValue(z)
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
        old_x = self.toy_list[self.current_toy]["obj"]["center"][0]
        self.toy_list[self.current_toy]["obj"]["item"].translate(value - old_x, 0, 0)
        self.toy_list[self.current_toy]["obj"]["center"][0] = value
        return

    def y_changed(self, box, value):
        old_x = self.toy_list[self.current_toy]["obj"]["center"][1]
        self.toy_list[self.current_toy]["obj"]["item"].translate(0, value - old_x, 0)
        self.toy_list[self.current_toy]["obj"]["center"][1] = value
        return

    def z_changed(self, box, value):
        old_x = self.toy_list[self.current_toy]["obj"]["center"][2]
        self.toy_list[self.current_toy]["obj"]["item"].translate(0, 0, value - old_x)
        self.toy_list[self.current_toy]["obj"]["center"][2] = value
        return

    def update_toy(self, pos):
        old_x, old_y, old_z = self.toy_list[self.current_toy]["obj"]["center"]
        self.toy_list[self.current_toy]["obj"]["item"].translate(pos[0]-old_x, pos[1]-old_y, pos[2]-old_z)
        self.toy_list[self.current_toy]["obj"]["center"] = pos
        self.max_XEdit.setValue(pos[0])
        self.max_YEdit.setValue(pos[1])
        self.max_ZEdit.setValue(pos[2])
        return

    def save_pos(self):
        if self.curr_indice == -1: return
        curr_frame = self.frame_list[self.curr_indice]
        toy = self.toy_list[self.current_toy]["obj"]

        self.ToyBox.model().item(self.current_toy).setBackground(Qt.green)
        name = self.toy_list[self.current_toy]["name"]
        if name not in self.viewer3D.room.toy_to_update:
            self.viewer3D.room.toy_to_update.append(name)
            
        toy["data"][curr_frame]["p3d"] = np.copy(toy["center"])
        
        self.frame_id.emit(curr_frame)
        if curr_frame not in self.corrected_list[self.current_toy]:
            self.corrected_list[self.current_toy].add(curr_frame)
        self.frame_listW.item(self.curr_indice).setBackground(Qt.green)
        self.project2D()
        return

    def project3D(self, data):
        if len(data) < 3: return
        if self.curr_indice == -1: return 
        att = to_3D(data, self.cams, self.h, self.w)  
        self.update_toy(att)
        pixels =  project_2d_simple(att, self.cams, self.h, self.w, is_mat = self.cam_id == 1)
        for k in pixels.keys(): pixels[k] = {"toy":pixels[k]}
        self.pose2d.emit(pixels)
        return

    def project2D(self, update_2d = False):    
        if self.curr_indice == -1: return
        item = self.toy_list[self.current_toy]["obj"]
        pixels = project_2d_simple(item["center"], self.cams, self.h, self.w, is_mat = self.cam_id == 1)
        for k in pixels.keys(): pixels[k] = {"toy":pixels[k]}
        self.pose2d.emit(pixels)
        return 

    def propagate(self, threshold = 30):
        if len(self.corrected_list[self.current_toy]) == 0: 
            print("No frame corrected")
            return 
        toy = self.toy_list[self.current_toy]["obj"]
        
        frame_list = sorted([x for x in toy["data"].keys() if "p3d" in toy["data"][x]])
        if len(frame_list) < 2: return
        
        corrected_list = np.array(sorted(self.corrected_list[self.current_toy]))
        to_keep = []
        print("In propagation", corrected_list)

        # First fill in missing data
        for i in range(len(corrected_list)-1):
            start = corrected_list[i]
            if start+1 in frame_list:
                to_keep.append(i)
                while start < corrected_list[i+1] and start+1 in frame_list:
                    start += 1
                if start == corrected_list[i+1]: continue
            end = corrected_list[i+1]
            while end-1 in frame_list:
                end -= 1
            poses = np.array([toy["data"][start]["p3d"], toy["data"][end]["p3d"]])
            interp_func = interpolate.interp1d(np.array([start,end]), poses, axis=0, kind = 'linear')    
            x_interp = interp_func(np.arange(start, end, 1))
            #print(corrected_list[i], start, end)
            for i, f in enumerate(range(start, end)):
                if f not in toy["data"]: toy["data"][f] = {}
                toy["data"][f]["p3d"] = x_interp[i, :3]

        if corrected_list[-1]-1 in frame_list or  corrected_list[-1]+1 in frame_list: to_keep.append(i+1)               
        if len(to_keep) == 0: return     
        corrected_list = corrected_list[to_keep]
        
        f = corrected_list[0]
        corrected_merged = [[f]]

        for f in corrected_list[1:]:
            prev = corrected_merged[-1][-1]
            if abs(f-prev) < threshold:
                corrected_merged[-1].append(f)
            else:
                corrected_merged.append([f])

        print(corrected_merged)        
        self.write_attention("temp_toy.npy")
        
        
        min_f, max_f = min(toy["data"].keys()), max(toy["data"].keys())
        for seg in corrected_merged: 
            seg = sorted(seg)        
            start, end = max(min_f, seg[0]-threshold), min(seg[-1] + threshold, max_f)
            while start not in toy["data"].keys():
                start += 1
            while end not in toy["data"].keys():
                end -= 1
                                         
            print(seg, start, end)
            mask = build_mask([x-start for x in seg], end-start+1, threshold = threshold)[:, np.newaxis]
            interp_list = np.unique([start] + seg + [end])
            interp_poses = []
            for f in interp_list:
                interp_poses.append(toy["data"][f]["p3d"]) 
            interp_poses = np.array(interp_poses)   
            if len(interp_list) == 2:
                interp_func = interpolate.interp1d(interp_list, interp_poses, axis=0, kind = 'linear')
            else: interp_func = interpolate.interp1d(interp_list, interp_poses, axis=0, kind = 'quadratic')
            x_interp = interp_func(np.arange(start, end, 1))
            old_p = None

            for i, f in enumerate(range(start, end)):
                p = toy["data"][f]["p3d"]
                m = mask[i]
                toy["data"][f]["p3d"] = (1-m)*p + x_interp[i, :3]*m
                if p is None and old_p is not None:
                    p = copy.deepcopy(old_p)
                    continue 
                old_p = p                  
        return 

    def runGP(self, param):
        toy = self.toy_list[self.current_toy]["obj"]
        if len(toy["data"]) < 60: return

        x_tr, frame_list = [], []
        for i, (f, p) in enumerate(toy["data"].items()):
            if self.segmentIndex == 0: 
                frame_list.append(f)
                x_tr.append(p)
            else:
                _, start, end = self.viewer3D.segment.current[self.segmentIndex-1]
                if start <= f <= end: 
                    frame_list.append(f)
                    x_tr.append(p)

        self.write_attention("temp_toy.npy")
        N = 60 if param == 0 else len(toy["data"]) // 1800
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
            self.corrected_list[self.current_toy] = set()
            return

        self.curr_indice = 0
        self.frame_listW.clear()
        for f, s in zip(self.frame_list, uncertain_scores):
            self.frame_listW.addItem(ListWidgetItem("{} - {:.2f}".format(f,s)))
        self.frame_listW.setCurrentRow(0)
        self.corrected_list[self.current_toy] = set()
        self.update_frame()
        return
    
    def showCorrected(self):
        message = ''
        if self.viewer3D.segment is None:
            message += "No segments\n"
        else:
            for i, (stype, s,e) in enumerate(self.viewer3D.segment):
                message += "Segment {} -> {} - {} ({})\n".format(i, s, e, stype)
        if len(self.history_corrected) == 0:
            message += "\nNo Corrected Frames"
        else:
            message += '\n{} Corrected Frames:\n'.format(len(self.history_corrected))
            message += ", ".join([ str(x) for x, y in self.history_corrected.items() if y == 1])
        print(message)
        QMessageBox.about(self, "Info Hands", message)
        return

    def write_attention(self, fileName = None):
        write_results_toy(self, fileName = fileName)
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


    def closeEvent(self, event):
        for n, obj in self.viewer3D.room.toy_objects.items():
            obj["item"].opts['drawEdges'] = False
        self.pose2d.emit({})
        self.open_id.emit(False)
        event.accept()