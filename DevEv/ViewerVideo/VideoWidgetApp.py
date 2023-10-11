from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import cv2
import sys
import os
from scipy.spatial.transform import Rotation  

from .VideoThreadApp import VideoThread


def rotation_matrix_from_vectors(a, b):
    # Normalize the input vectors
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = np.dot(a, b)

    if abs(c + 1.0) < 1e-6:
        # In this case, the vectors are exactly opposite, so we need a 180-degree rotation.
        # A 180-degree rotation matrix around any axis is -1 times the identity matrix.
        return -np.eye(3)

    if abs(c - 1.0) < 1e-6:
        # In this case, the vectors are already aligned, so no rotation is needed.
        return np.eye(3)

    v = np.cross(a, b)
    s = np.linalg.norm(v)

    # Skew-symmetric cross product matrix
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    if tdx != None and tdy != None:
        center_x = tdx
        center_y = tdy

    start_point = (center_x, center_y)

    M = Rotation.from_euler("xyz", np.array([yaw, pitch, roll]), degrees = True).as_matrix()
    # Define the 3D face direction vector (for example, in this case, it points right in the image plane)
    x = M[0] / np.linalg.norm(M[0])
    y = M[1] / np.linalg.norm(M[1])
    z = M[2] / np.linalg.norm(M[2])

    # Calculate the ending point based on the vector length (you may need to adjust the scaling)
    end_point_x = (start_point[0] + int(x[0] * size),
                start_point[1] + int(x[1] * size))
    end_point_y = (start_point[0] + int(y[0] * size),
                start_point[1] + int(y[1] * size))
    end_point_z = (start_point[0] + int(z[0] * size*2),
                start_point[1] + int(z[1] * size*2))

    # Draw the vector onto the background image
    thickness = 4  # Line thickness
    cv2.line(img, start_point, end_point_x, (0, 0, 255), thickness)
    cv2.line(img, start_point, end_point_y, (0, 255, 0), thickness)
    cv2.line(img, start_point, end_point_z, (255, 0, 0), thickness)
    return img

class VideoApp(QWidget):
    frame_id = pyqtSignal(int)
    annotations_id = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        self.disply_width = 600
        self.display_height = 550
        self.annotation_on = False
        #self.disply_width = 670
        #self.display_height = 540
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(self.disply_width, self.display_height)
        #self.image_label.setStyleSheet("border :3px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.mousePressEvent = self.video_clicked


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
        self.duration = 0
        self.width_video, self.height_video = 0, 0
        self.last_position = 0
        self.p2d = {}
        self.clicked_att = {}
        # connect its signal to the update_image slot

        # start the thread     
        self.view = [0]
        self.setThread()
        
    def setThread(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.frame_id.connect(self.update_text)
        return

    def select_view(self, img):
        h, w, _ = img.shape
        if len(self.p2d) > 0:
            for c, info in self.p2d.items():
                if type(c) != int: continue                
                if "att" in info:
                    img = cv2.circle(img, info["att"], radius=15, color= (0,0,255), thickness=8)
                if "head" in info:
                    img = cv2.circle(img, info["head"], radius=4, color= (255,0,0), thickness=10)
                    if "att" in info: img = cv2.line(img, info["head"], info["att"],  color= (0,0,255), thickness=4)
                    elif "angle" in info: 
                        yaw, pitch, roll = info["angle"]
                        img = draw_axis(img, yaw, pitch, roll, tdx=info["head"][0], tdy=info["head"][1])
                if "att_v" in info:
                    img = cv2.circle(img, info["att_v"], radius=4, color= (0,0,255), thickness=10)

                                
        if self.view[0] == 0: return img
        im = []
        for view in self.view:
            if view == 1: im.append(img[:h//4, :w//2])
            elif view == 2: im.append(img[:h//4, w//2:])
            elif view == 3: im.append(img[h//4:h//2, :w//2])
            elif view == 4: im.append(img[h//4:h//2, w//2:])
            elif view == 5: im.append(img[h//2:3*h//4, :w//2])
            elif view == 6: im.append(img[h//2:3*h//4, w//2:])
            elif view == 7: im.append(img[3*h//4:, :w//2])
            else: im.append(img[3*h//4:, w//2:])
        im = np.concatenate(im, axis=0)
        return im

    def set_file(self, filename):
        self.thread._run_flag = False
        self.duration, self.height_video, self.width_video = self.thread.set_file(filename)   
        self.setWindowTitle(os.path.basename(filename))  

    def setPosition(self, position):
        self.thread.position_flag = position
        second = position//self.thread.fps
        self.textLabel.setText("Time: {} mn {} \t-\t Frame: {}".format(second//60, second % 60, position))
        self.last_position = position
        return

    def update_last_image(self):
        self.thread.get_last_image()

    def showImage(self):
        self.thread.get_image(self.last_position)

    def stop_video(self):
        self.thread._run_flag = False

    def start_video(self):
        self.thread._run_flag = True
        self.thread.start()

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

    def close_thread(self):
        self.thread._run_flag = False
        self.thread.exit()
        self.thread.wait()
        return
        
    def video_clicked(self, event):
        if not self.annotation_on: return
        self.stop_video()
        x = event.pos().x()
        y = event.pos().y()
        w = self.image_label.pixmap().width()
        h = self.image_label.pixmap().height()
        # depending on what kind of value you like (arbitary examples)

        if x < w and y < h:
            data = get_cam(x/w, y/h, self.width_video, self.height_video, self.view)
            print("videoo clicked", data, x, y, h, w)
            for c, info in data.items():
                if c in self.clicked_att: del self.clicked_att[c]
                else: self.clicked_att[c] = info
        self.update_image_proj(self.clicked_att)
        return

    @pyqtSlot(dict)
    def update_image_proj(self, poses):
        self.stop_video()
        self.p2d = poses
        self.thread.get_image(self.last_position, emit_frame=False)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(int)
    def update_text(self, frame):
        second = frame//self.thread.fps
        self.textLabel.setText("Frame: {} \t Time: {} min {} s".format(frame, second//60, second % 60))
        self.frame_id.emit(frame)

    @pyqtSlot(bool)
    def set_annotation(self, state):
        self.annotation_on = state

    @pyqtSlot(bool)
    def send_annotation_head(self, state):
        self.clicked_att["type"] = "head"
        self.annotations_id.emit(self.clicked_att)
        self.clicked_att = {}

    @pyqtSlot(bool)
    def send_annotation_att(self, state):
        self.clicked_att["type"] = "att"
        self.annotations_id.emit(self.clicked_att)
        self.clicked_att = {}


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
        
def get_cam(x, y, width_video, height_video, view):
    if view[0] == 0:
        x_v, y_v = int(x*width_video), int(y*height_video)
        x_m, y_m = int(x*width_video), int(y*height_video)
        if x < 0.5 and y < 0.25: 
            c = 0
        elif x < 0.5 and 0.5 > y >= 0.25: 
            c = 2
            y_m = y_v - int(height_video//4)
        elif x >= 0.5 and y < 0.25: 
            c = 1
            x_m = x_v - int(width_video//2)
        elif x >= 0.5 and 0.5 > y >= 0.25: 
            c = 3
            y_m = y_v - int(height_video//4)
            x_m = x_v - int(width_video//2)
        elif x < 0.5 and 0.75 > y >= 0.5: 
            c = 4
            y_m = y_v - int(height_video//2)
        elif x >= 0.5 and 0.75 > y >= 0.5: 
            c = 5
            y_m = y_v - int(height_video//2)
            x_m = x_v - int(width_video//2)
        elif x < 0.5 and y >= 0.75: 
            c = 6
            y_m = y_v - int(3*height_video//4)
        elif x >= 0.5 and y >= 0.75: 
            c = 7
            y_m = y_v - int(3*height_video//4)
            x_m = x_v - int(width_video//2)

        return {c:{"att_v": [x_v, y_v], "att_p": [x_m, y_m]}}

    data = {}
    clicked = 0
    if len(view) == 2 and y > 0.5:  
        clicked = 1
        y -= 0.5
    y *= len(view)
    #for i, v in enumerate(view):
    c = view[clicked] - 1
    x_v, y_v = int(x*width_video//2), int(y*height_video//4)
    if c in [1,3,5,7]: x_v += int(width_video//2)
    if c in [2,3]: y_v += int(height_video//4)
    elif c in [4,5]: y_v += int(height_video//2)
    elif c in [6,7]: y_v += int(3*height_video//4)
    x_m, y_m = int(x*width_video//2), int(y*height_video//4)
    data[c] = {"att_v": [x_v, y_v], "att_p": [x_m, y_m]}
    print(c, data[c])
        
    return data

def main_video():
    app = QApplication(sys.argv)
    player = VideoWindow()
    #player.resize(640+520, 480)
    player.resize(640+520 , 480 )
    player.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main_video()



