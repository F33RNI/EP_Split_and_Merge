"""
 Licensed under the Unlicense License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://unlicense.org

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import threading

import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
import qimage2ndarray
import SAM_worker

PROJECT_FOLDER = 'EP_data'


class EngineeringProject(QtWidgets.QMainWindow):
    def __init__(self):
        super(EngineeringProject, self).__init__()
        uic.loadUi('gui_ep.ui', self)
        self.show()

        self.btn_camera_start.clicked.connect(self.camera_start)
        self.btn_camera_stop.clicked.connect(self.camera_stop)
        self.btn_image_save.clicked.connect(self.image_save)
        self.btn_image_load.clicked.connect(self.image_load)
        self.btn_split_and_merge.clicked.connect(self.split_and_merge)

        self.cv_cap = None
        self.current_frame = None
        self.loaded_image = None

        self.camera_running = False
        self.image_running = False

    def camera_start(self):
        self.camera_running = True
        if self.check_dshow.isChecked():
            self.cv_cap = cv2.VideoCapture(self.camera_id.value(), cv2.CAP_DSHOW)
        else:
            self.cv_cap = cv2.VideoCapture(self.camera_id.value())
        thread = threading.Thread(target=self.cv_thread)
        thread.start()

    def camera_stop(self):
        self.camera_running = False
        self.image_running = False

    def image_save(self):
        if self.camera_running:
            cv2.imwrite(PROJECT_FOLDER + '/image.png', self.current_frame)
        else:
            print('Camera not running!')

    def image_load(self):
        self.image_running = True
        self.loaded_image = cv2.imread(PROJECT_FOLDER + '/image.png')
        thread = threading.Thread(target=self.cv_thread)
        thread.start()

    def cv_thread(self):
        while self.camera_running or self.image_running:
            img = np.zeros((640, 480, 3), dtype=np.uint8)
            if self.camera_running:
                ret, img = self.cv_cap.read()
            elif self.image_running:
                img = self.loaded_image
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.current_frame)))

        self.cv_cap.release()

    def split_and_merge(self):
        self.loaded_image = cv2.imread(PROJECT_FOLDER + '/image.png')
        self.current_frame = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2GRAY)

        sam_worker = SAM_worker.SAM(self.current_frame)
        self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(sam_worker.image)))

        sam_worker.split_and_merge()

        self.cvl_image_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(sam_worker.debug)))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = EngineeringProject()
    window.show()
    app.exec_()
