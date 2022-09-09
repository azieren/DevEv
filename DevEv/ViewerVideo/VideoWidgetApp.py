from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import cv2
import sys

from .VideoThreadApp import VideoThread

class VideoApp(QWidget):
    frame_id = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        self.disply_width = 600
        self.display_height = 450
        #self.disply_width = 670
        #self.display_height = 540
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(self.disply_width, self.display_height)
        #self.image_label.setStyleSheet("border :3px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # create a text label
        self.textLabel = QLabel('Video')
        self.textLabel.setStyleSheet("border :1px solid black;")
        #self.image_label.setMaximumHeight(5)

        # create a vertical box layout and add the two labels

        
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)#, alignment=Qt.AlignCenter)
        vbox.addWidget(self.textLabel, alignment=Qt.AlignBottom)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        
        # create the video capture thread
        self.thread = VideoThread()
        self.duration = 0
        self.width_video, self.height_video = 0, 0
        self.last_position = 0
        self.p2d = {}
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.frame_id.connect(self.update_text)
        # start the thread     
        self.view = 0

    def select_view(self, img):
        h, w, _ = img.shape
        if len(self.p2d) > 0:
            for c, info in self.p2d.items():
                img = cv2.circle(img, info["att"], radius=20, color= (0,0,255), thickness=20)
                img = cv2.circle(img, info["head"], radius=5, color= (0,0,255), thickness=20)
                img = cv2.line(img, info["head"], info["att"],  color= (0,0,255), thickness=5)
        if self.view == 0: return img
        elif self.view == 1: return img[:h//2, :w//2]
        elif self.view == 2: return img[:h//2, w//2:]
        elif self.view == 3: return img[h//2:, :w//2]
        return img[h//2:, w//2:]

    def set_file(self, filename):
        self.thread.terminate()
        self.thread.wait()
        self.duration, self.height_video, self.width_video = self.thread.set_file(filename)
        self.thread.start()       

    def setPosition(self, position):
        self.thread.position_flag = position
        self.textLabel.setText(str(position))
        self.last_position = position
        return

    def showImage(self):
        self.thread.get_image(self.last_position)

    def stop_video(self):
        self.thread.terminate()
        self.thread.wait()
        self.thread._run_flag = False

    def start_video(self):
        self.thread._run_flag = True
        self.thread.start()

    def closeEvent(self, event):
        self.thread.terminate()
        self.thread.wait()
        self.thread.close()
        event.accept()

    @pyqtSlot(dict)
    def update_image_proj(self, poses):
        self.p2d = poses
        self.thread.get_image(self.last_position, emit_frame=False)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(int)
    def update_text(self, frame):
        self.textLabel.setText(str(frame))
        self.frame_id.emit(frame)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        cv_img = self.select_view(cv_img)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)

        #convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)
        

def main_video():
    app = QApplication(sys.argv)
    player = VideoWindow()
    #player.resize(640+520, 480)
    player.resize(640+520 , 480 )
    player.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main_video()



