from PyQt5.QtCore import QDir, Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLineEdit, QButtonGroup,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QRadioButton, QSplitter, QFrame, QCheckBox
from PyQt5.QtGui import QIcon, QIntValidator

import sys

from DevEv.Viewer3D.Viewer3DApp import View3D
from DevEv.ViewerVideo.VideoWidgetApp import VideoApp
from DevEv.ViewerCorrection.WidgetCorrection import CorrectionWindow

class VideoWindow(QMainWindow):

    def __init__(self, video_file=None, att_file=None, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("DevEnv") 
        self.move(200, 100)

        self.mediaPlayer = VideoApp()
        self.mediaPlayer.frame_id.connect(self.setPosition)

        # 3D view
        self.main3Dviewer = View3D()
        self.main3Dviewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Correction
        self.correctionWidget = CorrectionWindow(self.main3Dviewer)
        self.correctionWidget.frame_id.connect(self.setPosition)
        self.correctionWidget.pose2d.connect(self.mediaPlayer.update_image_proj)

        # Button
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

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

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open Video', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open video')
        openAction.triggered.connect(self.openFile)

        #
        openAtt = QAction(QIcon('open.png'), '&Open Attention', self)        
        openAtt.setShortcut('Ctrl+A')
        openAtt.setStatusTip('Open attention')
        openAtt.triggered.connect(self.openFileAtt)

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

        # Create exit action
        view0Action = QAction(QIcon('exit.png'), '&Select All Views', self)        
        #viewAction.setShortcut('Ctrl+R')
        view0Action.setStatusTip('Select all views')
        view0Action.triggered.connect(self.view0Select)

        # Create exit action
        view1Action = QAction(QIcon('exit.png'), '&Select View 1', self)        
        #viewAction.setShortcut('Ctrl+R')
        view1Action.setStatusTip('Select a view 1')
        view1Action.triggered.connect(self.view1Select)

        view2Action = QAction(QIcon('exit.png'), '&Select View 2', self)        
        #viewAction.setShortcut('Ctrl+R')
        view2Action.setStatusTip('Select a view 2')
        view2Action.triggered.connect(self.view2Select)

        view3Action = QAction(QIcon('exit.png'), '&Select View 3', self)        
        #viewAction.setShortcut('Ctrl+R')
        view3Action.setStatusTip('Select a view 3')
        view3Action.triggered.connect(self.view3Select)

        view4Action = QAction(QIcon('exit.png'), '&Select View 4', self)        
        #viewAction.setShortcut('Ctrl+R')
        view4Action.setStatusTip('Select a view 4')
        view4Action.triggered.connect(self.view4Select)

        correctAction = QAction(QIcon('exit.png'), '&Open correction tool', self)        
        #viewAction.setShortcut('Ctrl+R')
        correctAction.setStatusTip('Correction Tool')
        correctAction.triggered.connect(self.correctSelect)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(openAtt)
        fileMenu.addAction(exitAction)
        fileMenu.addAction(resetAction)

        viewMenu = menuBar.addMenu('&View')
        viewMenu.addAction(view0Action)
        viewMenu.addAction(view1Action)
        viewMenu.addAction(view2Action)
        viewMenu.addAction(view3Action)
        viewMenu.addAction(view4Action)

        correctMenu = menuBar.addMenu('&Correction')
        correctMenu.addAction(correctAction)

        vizLayout = QVBoxLayout()
        vizLayout.addWidget(self.showVecButton)
        vizLayout.addWidget(self.showHullButton)
        vizLayout.addWidget(self.addPCheck)
        vizLayout.addWidget(self.addFloorCheck)


        sceneBLayout = QVBoxLayout()
        sceneBLayout.addWidget(self.resetButton)
        sceneBLayout.addWidget(self.clearRoomButton)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addLayout(sceneBLayout)
        controlLayout.addWidget(self.showAllButton)

        showAllLayout = QVBoxLayout()
        showAllLayout.addWidget(self.minFrameEdit)
        showAllLayout.addWidget(self.maxFrameEdit)
        showAllLayout.addWidget(self.HeadCheck)
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

        splitter = QSplitter(Qt.Horizontal, frameShape=QFrame.StyledPanel,frameShadow=QFrame.Plain)
        splitter.setStyleSheet("QSplitter::handle{background: #444444;}") 
        splitter.addWidget(self.mediaPlayer)
        splitter.addWidget(view3Dwid)
        #splitter.setCollapsible(0, False)
        #splitter.setCollapsible(1, False)
        #splitter.setStretchFactor(0,0)

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(splitter)
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

    def setFile(self, filename):
        self.mediaPlayer.set_file(filename)
        self.playButton.setEnabled(True)
        self.positionSlider.setRange(0, self.mediaPlayer.duration)
        self.minInt.setTop(self.mediaPlayer.duration - 10)
        self.maxInt.setTop(self.mediaPlayer.duration)
        self.correctionWidget.setHW(self.mediaPlayer.height_video, self.mediaPlayer.width_video)       

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())
        if fileName != '':
            self.setFile(fileName)

    def openFileAtt(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Attention",
                QDir.homePath())

        if fileName != '':
            self.main3Dviewer.attention = self.main3Dviewer.read_attention(fileName)

    def view0Select(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.mediaPlayer.view = 0

    def view1Select(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.mediaPlayer.view = 1

    def view2Select(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.mediaPlayer.view = 2

    def view3Select(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.mediaPlayer.view = 3

    def view4Select(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.mediaPlayer.view = 4

    def correctSelect(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
        self.correctionWidget.show()
        self.correctionWidget.update_frame()

    def exitCall(self):
        self.mediaPlayer.close()
        self.close()

    def exitCall(self):
        self.mediaPlayer.close()
        self.close()

    def reset3D(self):
        self.main3Dviewer.reset()

    def play(self):
        if self.mediaPlayer.thread._run_flag:
            self.mediaPlayer.stop_video()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.mediaPlayer.start_video()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))            

    def sliderPause(self):
        # When slider is clicked
        self.mediaPlayer.stop_video()
        self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))

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
            if self.mediaPlayer.thread._run_flag:
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
       

def run(video_file=None, att_file=None):
    """
    run(video_file=None, att_file=None) function run the GUI for visualizaing video

    :video_file (Optionnal): video file to visualize, if nothing is provided the video wiget will be empty
    :att_file (Optionnal): attention file containing 3D data, if nothing is provided the 3D wigdet will just display the room
    :return: Nothing, the application ends when the GUI is closed
    """ 
    app = QApplication(sys.argv)
    player = VideoWindow(video_file=video_file, att_file=att_file)
    player.resize(640+520 , 480)
    player.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()



