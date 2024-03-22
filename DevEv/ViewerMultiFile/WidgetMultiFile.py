import os
import re
from PyQt5.QtWidgets import QStyle, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QDialog, QMessageBox, QListWidget, \
                                QGridLayout, QRadioButton, QListWidgetItem, QCheckBox, QComboBox, QScrollArea, QFileDialog
from PyQt5.QtCore import pyqtSignal, QDir, Qt
#from PyQt5.QtGui import QMessageBox
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

from DevEv.Viewer3D.utils import draw_cone, draw_Ncone

def parse_attention(path, start, end):
    head, att, handL, handR = [], [], [], []
    with open(path, "r") as f:
        data = f.readlines()
    for i, d in enumerate(data):
        d_split = d.replace("\n", "").split(",")
        xhl, yhl, zhl, xhr, yhr, zhr = 0,0,0,0,0,0
        if len(d_split)== 10:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d_split
        elif len(d_split)== 11:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2, flag = d_split
        elif len(d_split)== 18:
            frame, flag, flag_h, b0, b1, b2, A0, A1, A2, att0, att1, att2, xhl, yhl, zhl, xhr, yhr, zhr = d_split
        elif len(d_split) < 10: continue
        else:
            print("Error in attention file", path)
            break
        frame = int(frame)
        if frame < start: continue
        if frame > end: break
        a = np.array([float(att0), float(att1), float(att2)])
        h = np.array([float(b0), float(b1), float(b2)])
        hL = np.array([float(xhl), float(yhl), float(zhl)])
        hR = np.array([float(xhr), float(yhr), float(zhr)])

        size = np.linalg.norm(a - h)
        if size < 1e-6: 
            continue
        vec = (a - h)/ ( size + 1e-6)
        head.append(h)
        att.append(vec)
        handL.append(hL)
        handR.append(hR)

    return head, att, handL, handR           

class SegmentCheckBox(QCheckBox):
    def __init__(self, text='', data=None, parent=None):
        super().__init__(text, parent)
        self.data = data

    def get_custom_data(self):
        return self.data

    def set_custom_data(self, data):
        self.data = data


class MultiFileVisualizer(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, viewer3D):
        super().__init__()

        self.setWindowTitle("Multi File Visualizer") 
        self.resize(400 , 200 )
        self.viewer3D = viewer3D
        self.timestamps = self.viewer3D.segment.timestamps
        self.viz_flags = {"head":False, "att":False, "handL":False, "handR":False}
        self.category_mapping = {"r":0, "c":1, "p":2}
        self.directory_path = ""
        self.current_display = {}
        self.att_type = 0
        
        self.init_ui()
        self.initialize()

    def initialize(self):
        if "att_cone" in self.current_display: 
            self.viewer3D.removeItem(self.current_display["att_cone"])
            del self.current_display["att_cone"]
            
        u =  np.array([[0.0,0.0,0.0], [0.0,0.0,1.0]])
        self.current_display["head"] = gl.GLScatterPlotItem(pos = u[0].reshape(1,3), color=(0.0,0.0,1.0,1.0), size = np.array([8.0]),  glOptions = 'translucent')
        self.current_display["att_vec"] = gl.GLLinePlotItem(pos = u, color = (1.0,0.0,0.0,0.7), width= 3.0, antialias = True, glOptions = 'additive', mode = 'lines')
        self.current_display["att_cone"] = draw_cone(u[0], u[1], (1.0,0.0,0.0,0.7))
        self.current_display["handL"] = gl.GLScatterPlotItem(pos = u[0].reshape(1,3), color=(1.0,0.0,1.0,1.0), size = np.array([5.0]), glOptions = 'translucent')
        self.current_display["handR"] = gl.GLScatterPlotItem(pos = u[0].reshape(1,3), color=(0.0,1.0,0.0,1.0), size = np.array([5.0]), glOptions = 'translucent')
        for _, obj in self.current_display.items():
            if type(obj) == int or obj is None: continue
            obj.hide()
            self.viewer3D.addItem(obj)
        # Clear Flags
        for k in self.viz_flags.keys():
            self.viz_flags[k] = False
        # Clear checkbox
        self.headCheckBox.setChecked(False)
        self.attCheckBox.setChecked(False)
        self.handLCheckBox.setChecked(False)
        self.handRCheckBox.setChecked(False)
        self.clear_scrolls()
        self.directory_path = ""
        return

    def init_ui(self):
        self.select_directory_button = QPushButton("Open Directory")
        self.select_directory_button.clicked.connect(self.select_directory)

        self.display_button = QPushButton("Display")
        self.display_button.clicked.connect(self.display)
        
        self.headCheckBox = QCheckBox("&Head", self)
        self.headCheckBox.setChecked(False)
        self.headCheckBox.setEnabled(True)
        self.headCheckBox.clicked.connect(self.headCheck)

        self.attCheckBox = QCheckBox("&Att", self)
        self.attCheckBox.setChecked(False)
        self.attCheckBox.setEnabled(True)
        self.attCheckBox.clicked.connect(self.attCheck)

        self.handLCheckBox = QCheckBox("&Left Hand", self)
        self.handLCheckBox.setChecked(False)
        self.handLCheckBox.setEnabled(True)
        self.handLCheckBox.clicked.connect(self.handLCheck)

        self.handRCheckBox = QCheckBox("&Right Hand", self)
        self.handRCheckBox.setChecked(False)
        self.handRCheckBox.setEnabled(True)
        self.handRCheckBox.clicked.connect(self.handRCheck)
        
        
        self.vectorButton = QRadioButton("Vector")
        self.vectorButton.setChecked(True)
        self.vectorButton.toggled.connect(lambda:self.toggle_attention(self.vectorButton))
        self.coneButton = QRadioButton("Cone")
        self.coneButton.toggled.connect(lambda:self.toggle_attention(self.coneButton))
              
        buttonlayout = QHBoxLayout()
        buttonlayout.addWidget(self.select_directory_button)
        buttonlayout.addWidget(self.display_button)
        buttonlayout.addWidget(self.headCheckBox)
        buttonlayout.addWidget(self.attCheckBox)
        buttonlayout.addWidget(self.vectorButton)
        buttonlayout.addWidget(self.coneButton)
        buttonlayout.addWidget(self.handLCheckBox)
        buttonlayout.addWidget(self.handRCheckBox)

                
        # Names for the scroll areas
        names = ["Room Self Play", "Mat Self Play", "Parent Play"]
        # Create layout for labels and scroll areas
        labels_and_scrolls_layout = QGridLayout()
        self.scroll_layout = []
        # Add labels and scroll areas
        for i, name in enumerate(names):
            # Create label
            scroll_label = QLabel(name)

            # Create scroll area and layout
            scroll_area_layout = QVBoxLayout()
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_widget = QWidget()
            scroll_widget.setLayout(scroll_area_layout)
            scroll_area.setWidget(scroll_widget)

            # Add label and scroll area to layout
            labels_and_scrolls_layout.addWidget(scroll_label, 0, i)
            labels_and_scrolls_layout.addWidget(scroll_area, 1, i)
            self.scroll_layout.append(scroll_area_layout)

        layout = QVBoxLayout()
        # Add directory selection layout
        layout.addLayout(buttonlayout)
        layout.addLayout(labels_and_scrolls_layout)
        self.setLayout(layout)

    def clear_scrolls(self):
        # Clear existing checkboxes in all scroll areas
        for scroll_layout in self.scroll_layout:
            while scroll_layout.count():
                child = scroll_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        return

    def select_directory(self):
        directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            self.directory_path = directory_path
            self.populate_list()

    def populate_list(self):
        self.clear_scrolls()

        # Populate list with files in selected directory
        for file_name in os.listdir(self.directory_path):
            if os.path.isfile(os.path.join(self.directory_path, file_name)):
                file_info = self.get_file_info(file_name)
                if file_info is not None:
                    for ind, text in file_info.items():
                        scroll_layout = self.scroll_layout[ind]
                        file_checkbox = SegmentCheckBox(text[0], data = (text[1],text[2]))
                        file_label = QLabel("  Segment: {:d} - {:d}".format(text[1],text[2]))
                        scroll_layout.addWidget(file_checkbox)
                        scroll_layout.addWidget(file_label)

    def get_file_info(self, file_path):
        # Here you would implement your logic to determine which scroll area the file should belong to
        # For demonstration purposes, let's assume we extract a number from the file name and return it modulo 3
        # You should replace this with your actual logic
        if not file_path.startswith("attC_"): return None
        if not file_path.endswith(".txt"): return None
        sess_name = re.findall(r'\d\d_\d\d', file_path)
        if len(sess_name) == 0: return None
        data = self.timestamps[sess_name[0]]
        output = {}
        for k, list_seg in data.items():
            for (s, e) in list_seg:
                output[self.category_mapping[k]] = [file_path, s, e]
               
        return output

    def get_selected_files(self):
        selected_files = []
        for scroll_layout in self.scroll_layout:
            for i in range(scroll_layout.count()):
                widget = scroll_layout.itemAt(i).widget()
                if isinstance(widget, SegmentCheckBox) and widget.isChecked():
                    selected_files.append([widget.text(), widget.get_custom_data()])
        return selected_files
    
    def display(self):
        if "att_cone" in self.current_display: 
            self.viewer3D.removeItem(self.current_display["att_cone"])
            del self.current_display["att_cone"]
            
        filelist = self.get_selected_files()
        head_all, att_all, handL_all, handR_all = [], [], [], []
        for fileinfo in filelist:
            filepath = fileinfo[0]
            start, end = fileinfo[1]
            path = os.path.join(self.directory_path, filepath)
            head, att, handL, handR = parse_attention(path, start, end)
            head_all.extend(head)
            att_all.extend(att)
            handL_all.extend(handL)
            handR_all.extend(handR)
        if len(head_all) < 2: return
        att_all, head_all = np.array(att_all).reshape(-1,3), np.array(head_all).reshape(-1,3)
        handL_all, handR_all = np.array(handL_all).reshape(-1,3), np.array(handR_all).reshape(-1,3)
        # Subsample:
        att_all, head_all, handL_all, handR_all = att_all[::2], head_all[::2], handL_all[::2], handR_all[::2]
        
        vec = np.stack([head_all, head_all + att_all], axis=1).reshape(-1,3)
        self.current_display["head"].setData(pos = head_all)
        self.current_display["att_vec"].setData(pos = vec)
        d, _ = draw_Ncone(head_all, head_all + att_all)
        self.current_display["att_cone"] = gl.GLMeshItem(meshdata=d, color = (1.0,0.0,0.0,0.7), glOptions = 'translucent', drawEdges=True, antialias=True, computeNormals=False)
        self.current_display["handL"].setData(pos = handL_all)
        self.current_display["handR"].setData(pos = handR_all)
        self.viewer3D.addItem(self.current_display["att_cone"])
        self.attCheck(self.viz_flags["att"])
        return
    
    def headCheck(self, state):
        self.viz_flags["head"] = state
        self.current_display["head"].setVisible(state)

    def attCheck(self, state):
        self.viz_flags["att"] = state
        if self.att_type == 0:
            self.current_display["att_vec"].setVisible(state)
            self.current_display["att_cone"].setVisible(False)
        else:
            self.current_display["att_vec"].setVisible(False)
            self.current_display["att_cone"].setVisible(state)
            
    def handLCheck(self, state):
        self.viz_flags["handL"] = state
        self.current_display["handL"].setVisible(state)
   
    def handRCheck(self, state):
        self.viz_flags["handR"] = state
        self.current_display["handR"].setVisible(state)
        
    def closeEvent(self, event):
        event.accept()
    
    def toggle_attention(self, b):
        if b.text() == "Vector":
            if b.isChecked() == True:
                self.att_type = 0	
        else:
            if b.isChecked() == True:
                self.att_type = 1
        self.attCheck(self.viz_flags["att"])
        return
    