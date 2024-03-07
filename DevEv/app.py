from PyQt5.QtCore import QDir, Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLineEdit, QButtonGroup,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QRadioButton, QSplitter, QFrame, QCheckBox
from PyQt5.QtGui import QIcon, QIntValidator

import sys

from DevEv.Viewer3D.Viewer3DApp import View3D
from DevEv.ViewerVideo.VideoWidgetApp import VideoApp
from DevEv.ViewerCorrection.WidgetCorrection import CorrectionWindow
from DevEv.ViewerCorrection.WidgetCorrectionHand import CorrectionWindowHand
from DevEv.ViewerCorrection.WidgetCorrectionToys import CorrectionWindowToys

class VideoWindow(QMainWindow):

    def __init__(self, video_file=None, att_file=None, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("DevEv") 
        self.move(200, 100)

        self.mediaPlayer = VideoApp()
        self.mediaPlayer.frame_id.connect(self.setPosition)

        # 3D view
        self.main3Dviewer = View3D()
        self.main3Dviewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Correction
        self.correctionWidget = CorrectionWindow(self.main3Dviewer)
        self.correctionWidget.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.correctionWidget.frame_id.connect(self.setPosition)
        self.correctionWidget.pose2d.connect(self.mediaPlayer.update_image_proj)
        self.correctionWidget.open_id.connect(self.mediaPlayer.set_annotation)
        self.correctionWidget.open_id.connect(self.main3Dviewer.set_annotation)
        self.correctionWidget.project3dButtonAtt.clicked.connect(self.mediaPlayer.send_annotation_att)
        self.correctionWidget.project3dButtonHead.clicked.connect(self.mediaPlayer.send_annotation_head)
        self.mediaPlayer.annotations_id.connect(self.correctionWidget.project3D)

        self.correctionWidgetHands = CorrectionWindowHand(self.main3Dviewer)
        self.correctionWidgetHands.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.correctionWidgetHands.frame_id.connect(self.setPosition)
        self.correctionWidgetHands.pose2d.connect(self.mediaPlayer.update_image_proj)
        self.correctionWidgetHands.open_id.connect(self.mediaPlayer.set_annotation)
        self.correctionWidgetHands.open_id.connect(self.main3Dviewer.set_annotation)        
        self.correctionWidgetHands.project3dButtonLeft.clicked.connect(self.mediaPlayer.send_annotation_handL)
        self.correctionWidgetHands.project3dButtonRight.clicked.connect(self.mediaPlayer.send_annotation_handR)
        self.mediaPlayer.annotations_id.connect(self.correctionWidgetHands.project3D)


        self.correctionWidgetToys = CorrectionWindowToys(self.main3Dviewer)
        self.correctionWidgetToys.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.correctionWidgetToys.pose2d.connect(self.mediaPlayer.update_image_proj)
        self.correctionWidgetToys.frame_id.connect(self.setPosition)
        self.correctionWidgetToys.open_id.connect(self.mediaPlayer.set_annotation)
        self.correctionWidgetToys.open_id.connect(self.main3Dviewer.set_annotation)     
        self.correctionWidgetToys.project3dButton.clicked.connect(self.mediaPlayer.send_annotation_toy)   
        self.mediaPlayer.annotations_id.connect(self.correctionWidgetToys.project3D)   

        # Button
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.playBackButton = QPushButton()
        self.playBackButton.setEnabled(False)
        self.playBackButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.playBackButton.clicked.connect(self.playback)

        self.playFrontButton = QPushButton()
        self.playFrontButton.setEnabled(False)
        self.playFrontButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.playFrontButton.clicked.connect(self.playfront)


        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderPressed.connect(self.sliderPause)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.positionSlider.sliderReleased.connect(self.setImageSlider)

                
        self.resetButton = QPushButton("&Reset View", self)
        self.resetButton.setEnabled(True)
        self.resetButton.clicked.connect(self.reset3D)

        self.clearRoomButton = QCheckBox("&Hide scene", self)
        self.clearRoomButton.setEnabled(True)
        self.clearRoomButton.setChecked(False)
        self.clearRoomButton.clicked.connect(self.main3Dviewer.clearRoom)

        self.showAllButton = QCheckBox("&Show Range", self)
        self.showAllButton.setChecked(False)
        self.showAllButton.clicked.connect(self.showAll)

        self.group_viz = QButtonGroup()
        self.group_viz.setExclusive(True)  # Radio buttons are not exclusive
        self.group_viz.buttonClicked.connect(self.toggle_viz)
        self.showVecButton = QRadioButton("Vector")
        self.showVecButton.setChecked(True)
        self.showVecButton.setEnabled(False)
        self.group_viz.addButton(self.showVecButton)

        self.showHullButton = QRadioButton("Hull")
        self.showHullButton.setEnabled(False)
        self.group_viz.addButton(self.showHullButton)

        self.minFrameEdit = QLineEdit("0", self)
        self.minInt = QIntValidator(0, 999999, self)
        self.minFrameEdit.setValidator(self.minInt)
        self.minFrameEdit.setEnabled(False)
        self.minFrameEdit.setMaximumWidth(70)
        self.minFrameEdit.setFixedWidth(60)
        self.minFrameEdit.editingFinished.connect(self.changeShowAllmin)

        self.maxFrameEdit = QLineEdit("1", self)
        self.maxInt = QIntValidator(1, 999999, self)
        self.maxFrameEdit.setValidator(self.maxInt)
        self.maxFrameEdit.setEnabled(False)
        self.maxFrameEdit.setMaximumWidth(70)
        self.maxFrameEdit.setFixedWidth(60)
        self.maxFrameEdit.editingFinished.connect(self.changeShowAllmax)

        self.vectorButton = QRadioButton("Vector")
        self.vectorButton.setChecked(True)
        self.vectorButton.toggled.connect(lambda:self.toggle_attention(self.vectorButton))
        self.lineButton = QRadioButton("Line")
        self.lineButton.toggled.connect(lambda:self.toggle_attention(self.lineButton))
        self.coneButton = QRadioButton("Cone")
        self.coneButton.toggled.connect(lambda:self.toggle_attention(self.coneButton))
        self.noneButton = QRadioButton("None")
        self.noneButton.toggled.connect(lambda:self.toggle_attention(self.noneButton))

        self.checkbox = QCheckBox("&Accumulate Attention", self)
        self.checkbox.clicked.connect(self.main3Dviewer.accumulate3D)

        self.fillUpBox = QCheckBox("&Fill", self)
        self.fillUpBox.setChecked(False)
        self.fillUpBox.clicked.connect(self.main3Dviewer.fill_acc)

        self.colorCheck = QCheckBox("&Time Colors", self)
        self.colorCheck.setChecked(False)
        self.colorCheck.clicked.connect(lambda:self.toggle_color(self.colorCheck))
        self.densityCheck = QCheckBox("&Density Colors", self)
        self.densityCheck.setChecked(False)
        self.densityCheck.clicked.connect(lambda:self.toggle_color(self.densityCheck))

        self.addPCheck = QCheckBox("&Attention", self)
        self.addPCheck.setChecked(False)
        self.addPCheck.setEnabled(False)
        self.addPCheck.clicked.connect(self.main3Dviewer.addPCheck)

        self.addFloorCheck = QCheckBox("&2D", self)
        self.addFloorCheck.setChecked(False)
        self.addFloorCheck.setEnabled(False)
        self.addFloorCheck.clicked.connect(self.floorCheck)

        self.HeadCheck = QCheckBox("&Head", self)
        self.HeadCheck.setChecked(True)
        self.HeadCheck.setEnabled(True)
        self.HeadCheck.clicked.connect(self.main3Dviewer.addHeadCheck)

        self.HandsCheck = QCheckBox("&Hands", self)
        self.HandsCheck.setChecked(False)
        self.HandsCheck.setEnabled(True)
        self.HandsCheck.clicked.connect(self.main3Dviewer.addHandCheck)
        
        self.speedCheckBox = QCheckBox("Speed x2", self)
        self.speedCheckBox.clicked.connect(self.speedUp)


        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open Video', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open video file')
        openAction.triggered.connect(self.openFile)

        #
        openAtt = QAction(QIcon('open.png'), '&Open Attention', self)        
        openAtt.setShortcut('Ctrl+A')
        openAtt.setStatusTip('Open attention file')
        openAtt.triggered.connect(lambda checked, param=False: self.openFileAtt(param) )

        openAtt2 = QAction(QIcon('open.png'), '&Open Attention as new', self)        
        openAtt2.setShortcut('Ctrl+N')
        openAtt2.setStatusTip('Open attention file without history')
        openAtt2.triggered.connect(lambda checked, param=True: self.openFileAtt(param) )

        openToy = QAction(QIcon('open.png'), '&Open Toy file', self)        
        openToy.setShortcut('Ctrl+L')
        openToy.setStatusTip('Opentoy file')
        openToy.triggered.connect(lambda checked, param=True: self.openFileToys(param) )


        openKpt = QAction(QIcon('open.png'), '&Open Keypoint', self)        
        openKpt.setShortcut('Ctrl+K')
        openKpt.setStatusTip('Open Keypoint file')
        openKpt.triggered.connect(self.openKptAtt)
        
        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create exit action
        resetAction = QAction(QIcon('exit.png'), '&Reset', self)        
        resetAction.setShortcut('Ctrl+R')
        resetAction.setStatusTip('Reset 3D View')
        resetAction.triggered.connect(self.reset3D)

        self.viewAction, self.curr_views = [], [0]
        for i in range(9):
            # Create exit action
            action = QAction('&Select View '+str(i), self, checkable=True)        
            action.setShortcut(str(i))
            if i == 0: action.setChecked(True)
            else: action.setChecked(False)
            action.setStatusTip('Select view '+str(i))
            action.setData(i)
            action.triggered.connect(self.viewSelect)
            self.viewAction.append(action)

        correctAction = QAction(QIcon('exit.png'), '&Att correction tool', self)        
        correctAction.setShortcut('Ctrl+C')
        correctAction.setStatusTip('Correction Tool Attention')
        correctAction.triggered.connect(self.correctSelect)

        correctHAction = QAction(QIcon('exit.png'), '&Hands correction tool', self)        
        correctHAction.setShortcut('Ctrl+H')
        correctHAction.setStatusTip('Correction Tool Hands')
        correctHAction.triggered.connect(self.correctHSelect)

        correctToyAction = QAction(QIcon('exit.png'), '&Toys correction tool', self)        
        correctToyAction.setShortcut('Ctrl+T')
        correctToyAction.setStatusTip('Correction Tool Toys')
        correctToyAction.triggered.connect(self.correctToySelect)
                
        self.roomActions = []
        titles = ['&Hide Room', '&Wireframe Room', '&Transparent Room', '&Solid Room']
        tips = ['Hide 3D Room', 'Show 3D Room in wireframe', 'Show 3D Room with transparence', 'Show 3D Room']
        for i in range(4):
            action = QAction(titles[i], self, checkable=True)        
            #action.setShortcut(str(i))
            if i == 3: action.setChecked(True)
            else: action.setChecked(False)
            action.setStatusTip(tips[i])
            action.setData(i)
            action.triggered.connect(self.toggleRoomStyle)
            self.roomActions.append(action)

        load2dAction = QAction(QIcon('exit.png'), '&Compute/Load 2D', self)        
        load2dAction.setShortcut('Ctrl+L')
        load2dAction.setStatusTip('Compute 2D info from attention file')
        load2dAction.triggered.connect(self.compute2D)
        
        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(openAtt)
        fileMenu.addAction(openAtt2)
        fileMenu.addAction(openToy)
        fileMenu.addAction(openKpt)
        fileMenu.addAction(resetAction)
        fileMenu.addAction(load2dAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)
        
        viewMenu = menuBar.addMenu('&View')
        for i in range(9):
            viewMenu.addAction(self.viewAction[i])
            if i == 0:
                viewMenu.addSeparator()

        correctMenu = menuBar.addMenu('&Correction')
        correctMenu.addAction(correctAction)
        correctMenu.addAction(correctHAction)
        correctMenu.addAction(correctToyAction)

        roomMenu = menuBar.addMenu('&Room View')
        for a in self.roomActions:
            roomMenu.addAction(a)
            
        self.camActions = []
        titles = ['&Room View', '&Mat View']
        tips = ['Setup the cameras on the entire room', 'Setup the cameras focus on the mat']
        for i in range(2):
            action = QAction(titles[i], self, checkable=True)        
            #action.setShortcut(str(i))
            if i == 0: action.setChecked(True)
            else: action.setChecked(False)
            action.setStatusTip(tips[i])
            action.setData(i)
            action.triggered.connect(self.toggleCams)
            self.camActions.append(action)         
        camMenu = menuBar.addMenu('&Select Cameras')
        for a in self.camActions:
            camMenu.addAction(a)
 
        self.view3DAction = []
        for i in range(8):
            # Create exit action
            action = QAction('&Cam '+str(i+1), self)        
            action.setStatusTip('Set 3D view to camera '+str(i+1))
            action.setData(i)
            action.triggered.connect(self.view3DSelect)
            self.view3DAction.append(action)
                       
        view3DMenu = menuBar.addMenu('&Set 3D View')
        for i in range(8):
            view3DMenu.addAction(self.view3DAction[i])
           
        vizLayout = QVBoxLayout()
        vizLayout.addWidget(self.showVecButton)
        vizLayout.addWidget(self.showHullButton)
        vizLayout.addWidget(self.addPCheck)
        vizLayout.addWidget(self.addFloorCheck)

        sceneBLayout = QVBoxLayout()
        sceneBLayout.addWidget(self.speedCheckBox)
        sceneBLayout.addWidget(self.resetButton)
        sceneBLayout.addWidget(self.clearRoomButton)

        controlVidLayout = QHBoxLayout()
        controlVidLayout.addWidget(self.playBackButton)
        controlVidLayout.addWidget(self.playButton)
        controlVidLayout.addWidget(self.playFrontButton)
        
        controlVid2Layout = QVBoxLayout()
        controlVid2Layout.addWidget(self.speedCheckBox)
        controlVid2Layout.addLayout(controlVidLayout)
        
        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addLayout(controlVid2Layout)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addLayout(sceneBLayout)
        controlLayout.addWidget(self.showAllButton)

        showAllLayout = QVBoxLayout()
        showAllLayout.addWidget(self.minFrameEdit)
        showAllLayout.addWidget(self.maxFrameEdit)
        showAllLayout.addWidget(self.HeadCheck)
        showAllLayout.addWidget(self.HandsCheck)
        controlLayout.addLayout(showAllLayout)
        controlLayout.addLayout(vizLayout)

        control3DLayout = QHBoxLayout()
        control3DLayout.addWidget(self.checkbox)
        control3DLayout.addWidget(self.fillUpBox)
        control3DLayout.addWidget(self.noneButton)
        control3DLayout.addWidget(self.vectorButton)#, alignment=Qt.AlignBottom)
        control3DLayout.addWidget(self.lineButton)
        control3DLayout.addWidget(self.coneButton)
        control3DLayout.addWidget(self.colorCheck)
        control3DLayout.addWidget(self.densityCheck)

        view3DLayout = QVBoxLayout()
        view3DLayout.addWidget(self.main3Dviewer)
        view3DLayout.addLayout(control3DLayout)

        view3Dwid = QWidget()
        view3Dwid.setLayout(view3DLayout)

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(view3Dwid)
        mainlayout.addLayout(controlLayout)

        # Set widget to contain window contents
        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(mainlayout)

        if video_file is not None:
            self.setFile(video_file)
            
        if att_file is not None:
            self.main3Dviewer.attention = self.main3Dviewer.read_attention(att_file)
            
    def playback(self):
        self.sliderPause()
        position = max(0, self.positionSlider.value() - 5)
        self.setPosition(position)
        self.mediaPlayer.showImage()
        return

    def playfront(self):
        self.sliderPause()
        position = min(self.mediaPlayer.duration, self.positionSlider.value() + 5)
        self.setPosition(position)
        self.mediaPlayer.showImage()
        return

    def setFile(self, filename):
        self.mediaPlayer.set_file(filename)
        self.playButton.setEnabled(True)
        self.playBackButton.setEnabled(True)
        self.playFrontButton.setEnabled(True)
        self.positionSlider.setRange(0, self.mediaPlayer.duration)
        

        
        self.minInt.setTop(self.mediaPlayer.duration - 10)
        self.maxInt.setTop(self.mediaPlayer.duration)
        self.correctionWidget.setHW(self.mediaPlayer.height_video, self.mediaPlayer.width_video)      
        self.correctionWidgetHands.setHW(self.mediaPlayer.height_video, self.mediaPlayer.width_video)  
        self.correctionWidgetToys.setHW(self.mediaPlayer.height_video, self.mediaPlayer.width_video)  
        self.mediaPlayer.showImage() 

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video",
                QDir.currentPath(),  "Video Files (*.avi *.mp4)" )#, options=QFileDialog.DontUseNativeDialog)
        
        if fileName != '':
            self.setFile(fileName)
            if not self.mediaPlayer.isVisible():
                self.mediaPlayer.show()

    def openFileAtt(self, as_new=False):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Attention",
                QDir.currentPath(), "Text files (*.txt)")#, options=QFileDialog.DontUseNativeDialog)
        
        if fileName != '':
            self.main3Dviewer.attention = self.main3Dviewer.read_attention(fileName, as_new=as_new)
            self.correctionWidget.update_list_frames()
            self.correctionWidgetHands.update_list_frames()
            #self.compute2D()

    def openFileToys(self, as_new=False):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Toy File",
                QDir.currentPath(), "Numpy files (*.npy)")#, options=QFileDialog.DontUseNativeDialog)
        
        if fileName != '':
            self.main3Dviewer.attention = self.main3Dviewer.read_toys(fileName, as_new=as_new)
            #self.compute2D()            

    def compute2D(self):        
        self.mediaPlayer.compute2D(self.main3Dviewer.attention, self.correctionWidget.cams)

    def openKptAtt(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Keypoint file",
                QDir.currentPath(), "NPY files (*.npy)")#, options=QFileDialog.DontUseNativeDialog)

        if fileName != '':
            self.main3Dviewer.attention = self.main3Dviewer.read_keypoints(fileName)


    def viewSelect(self):
        self.mediaPlayer.stop_video()
        view_id = self.sender().data()
        if view_id == 0:
            self.curr_views = [view_id]
        else:
            if len(self.curr_views) == 1:
                if self.curr_views[0] == 0:
                    self.curr_views = [view_id]
                elif view_id not in self.curr_views: self.curr_views.append(view_id)
            else:
                if view_id not in self.curr_views: 
                    self.curr_views.pop(0)
                    self.curr_views.append(view_id)
                else: self.curr_views.remove(view_id)

        for i in range(9): 
            if i in self.curr_views: self.viewAction[i].setChecked(True)
            else: self.viewAction[i].setChecked(False)
        self.mediaPlayer.view = sorted(self.curr_views)
        self.mediaPlayer.update_last_image()

    def view3DSelect(self):
        view_id = self.sender().data()
        cam_type = 1
        if self.camActions[0].isChecked(): cam_type = 0
        self.main3Dviewer.set3DView(view_id, cam_type)


    def toggleRoomStyle(self):
        view_id = self.sender().data()
        for i in range(len(self.roomActions)):
            self.roomActions[i].setChecked(False)
        self.roomActions[view_id].setChecked(True)
        self.main3Dviewer.setRoomStyle(view_id)

    def toggleCams(self):
        cam_id = self.sender().data()
        self.camActions[1-cam_id].setChecked(False)
        self.camActions[cam_id].setChecked(True)
        self.correctionWidget.setCams(cam_id)
        self.correctionWidgetHands.setCams(cam_id)
        self.correctionWidgetToys.setCams(cam_id)
        
    def correctSelect(self):
        self.mediaPlayer.stop_video()
        self.correctionWidget.show()
        self.correctionWidget.raise_()
        self.correctionWidget.update_frame()
        self.mediaPlayer.set_annotation(True)
        self.main3Dviewer.set_annotation(True)
        self.mediaPlayer.viz_flags["att"] = True
        self.mediaPlayer.viz_flags["head"] = True
        self.mediaPlayer.headCheckBox.setChecked(True)
        self.mediaPlayer.attCheckBox.setChecked(True)

    def correctHSelect(self):
        self.mediaPlayer.stop_video()
        self.correctionWidgetHands.show()
        self.correctionWidgetHands.raise_()
        self.correctionWidgetHands.update_frame()
        self.mediaPlayer.set_annotation(True)
        self.main3Dviewer.set_annotation(True)
        self.mediaPlayer.viz_flags["handL"] = True
        self.mediaPlayer.viz_flags["handR"] = True
        self.mediaPlayer.handLCheckBox.setChecked(True)
        self.mediaPlayer.handRCheckBox.setChecked(True)

    def correctToySelect(self):
        self.mediaPlayer.stop_video()
        self.correctionWidgetToys.show()
        self.correctionWidgetToys.raise_()
        self.correctionWidgetToys.update_frame()
        self.correctionWidgetToys.update_combo_toy()
        self.mediaPlayer.set_annotation(True)
        self.main3Dviewer.set_annotation(True)
                        
    def exitCall(self):
        self.close()

    def closeEvent(self, event):
        self.mediaPlayer.close()
        self.mediaPlayer.close_thread()
        self.correctionWidget.close()
        self.correctionWidgetHands.close()
        self.correctionWidgetToys.close()
        event.accept()


    def reset3D(self):
        self.main3Dviewer.reset()

    def play(self):
        if self.mediaPlayer.thread is None: return
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.mediaPlayer.start_video()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))            

    def sliderPause(self):
        # When slider is clicked
        self.mediaPlayer.stop_video()
        self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def setImageSlider(self):
        # Update Image after slider release
        self.mediaPlayer.showImage()
        return

    def setPosition(self, position):
        # When slider is moved
        self.positionSlider.setValue(position)
        self.mediaPlayer.setPosition(position)
        self.main3Dviewer.draw_frame(position, plot_vec = True)

    def showAll(self, state):
        self.minFrameEdit.setEnabled(state)
        self.maxFrameEdit.setEnabled(state)
        self.showVecButton.setEnabled(state)
        self.showHullButton.setEnabled(state)
        self.addPCheck.setEnabled(state)
        self.addFloorCheck.setEnabled(state)
        if state: 
            self.mediaPlayer.stop_video()
            minf, maxf = eval(self.minFrameEdit.text()), eval(self.maxFrameEdit.text())
            if maxf <= minf: maxf = minf + 1
            if self.showVecButton.isChecked():
                self.main3Dviewer.showAll(minf, maxf, 0)
            else:
                self.main3Dviewer.showAll(minf, maxf, 1)
        else:
            self.main3Dviewer.clear_t()

    def changeShowAllmin(self):
        minf, maxf = eval(self.minFrameEdit.text()), eval(self.maxFrameEdit.text())
        if maxf <= minf: 
            self.maxFrameEdit.setText(str(minf + 1))
        if self.showVecButton.isChecked():
            self.main3Dviewer.showAll(minf, maxf, 0)
        else:
            self.main3Dviewer.showAll(minf, maxf, 1)
        self.setPosition(minf)
        self.setImageSlider()
        return

    def changeShowAllmax(self):
        minf, maxf = eval(self.minFrameEdit.text()), eval(self.maxFrameEdit.text())
        if maxf <= minf: 
            self.minFrameEdit.setText(str(max(0, maxf - 1)))           
            self.setPosition(max(0, maxf - 1))
            self.setImageSlider()
        if self.showVecButton.isChecked():
            self.main3Dviewer.showAll(minf, maxf, 0)
        else:
            self.main3Dviewer.showAll(minf, maxf, 1)
        return

    def toggle_attention(self, b):

        if b.text() == "Vector":
            if b.isChecked() == True:
                self.main3Dviewer.line_type = 0	
        if b.text() == "Line":
            if b.isChecked() == True:
                self.main3Dviewer.line_type = 1
        if b.text() == "Cone":
            if b.isChecked() == True:
                self.main3Dviewer.line_type = 2
        if b.text() == "None":
            if b.isChecked() == True:
                self.main3Dviewer.line_type = 3
        self.main3Dviewer.draw_frame(None, plot_vec = True)
        if self.showAllButton.isChecked():
            self.changeShowAllmax()
        else:
            self.setPosition(self.positionSlider.value())
        return

    def toggle_viz(self, b):
        minf, maxf = eval(self.minFrameEdit.text()), eval(self.maxFrameEdit.text())
        if b.text() == "Vector":
            if b.isChecked() == True:
                self.main3Dviewer.showAll(minf, maxf, 0)
        if b.text() == "Hull":
            if b.isChecked() == True:
                self.main3Dviewer.showAll(minf, maxf, 1)
        return

    def toggle_color(self, b):
        if "Time" in b.text():
            if b.isChecked() == True:
                self.densityCheck.setChecked(False)
                self.main3Dviewer.colorCheck(1)
            else:
                self.main3Dviewer.colorCheck(0)
        if "Density" in b.text():
            if b.isChecked() == True:
                self.colorCheck.setChecked(False)
                self.main3Dviewer.colorCheck(2)
            else:
                self.main3Dviewer.colorCheck(0)
        self.showAll(self.showAllButton.isChecked())
        return

    def floorCheck(self, state):       
        self.main3Dviewer.project_floor = state     
        self.showAll(True)     
        return
    
    def speedUp(self, state):
        self.mediaPlayer.setSpeedUp(state)
        return
       

def run(video_file=None, att_file=None):
    """
    run(video_file=None, att_file=None) function run the GUI for visualizaing video

    :video_file (Optionnal): video file to visualize, if nothing is provided the video wiget will be empty
    :att_file (Optionnal): attention file containing 3D data, if nothing is provided the 3D wigdet will just display the room
    :return: Nothing, the application ends when the GUI is closed
    """ 
    app = QApplication(sys.argv)
    player = VideoWindow(video_file=video_file, att_file=att_file)
    player.resize(640+520 , 680)
    player.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()



