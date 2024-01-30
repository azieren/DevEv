from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
import cv2
import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    frame_id = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.cap = None
        self.curr_frame = 0
        self.fps = 1

    def set_file(self, filename):
        if self.cap is not None:
            self.cap.release() 
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.last_image = None
        self.curr_frame = 0
        self._run_flag = False
        self.position_flag = None
        self.duration = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_video = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_video = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.speedup = False
        return self.duration, height_video, width_video

    def run(self):
        # capture from web cam
        while 1:
            if self.cap is None: break

            if self._run_flag:
                ret, cv_img = self.cap.read()
                if self.speedup:
                    self.curr_frame += 1
                    ret, cv_img = self.cap.read()    
                if ret:   
                    self.last_image = cv_img            
                    self.change_pixmap_signal.emit(cv_img)                   
                    time.sleep(1/(self.fps+100))
                    self.curr_frame += 1
                else:
                    self.cap = cv2.VideoCapture(self.filename)
                    self.curr_frame = 0
                self.frame_id.emit(self.curr_frame)
            else: break
        return

    def get_last_image(self, emit_frame=True):
        if self.cap is None or self.last_image is None: return
        self.change_pixmap_signal.emit(self.last_image)
        if emit_frame: self.frame_id.emit(self.curr_frame)

    def get_image(self, position, emit_frame=True):
        if self.cap is None: return
        if 0 <= position < self.duration:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.curr_frame = position
            ret, cv_img = self.cap.read()
            self.last_image = cv_img
            if not ret: return
            self.change_pixmap_signal.emit(cv_img)
            if emit_frame: self.frame_id.emit(self.curr_frame)

    def setSpeedUp(self, state):
        self.speedup = state

    def close(self):
        self._run_flag = False
        if self.cap is not None:
            self.cap.release() 
        





