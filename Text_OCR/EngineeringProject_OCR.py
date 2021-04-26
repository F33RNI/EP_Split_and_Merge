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

import os
import threading

import numpy as np
import cv2
import sys

import qimage2ndarray
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
import random
import json
import os.path
from pathlib import Path

import ImageSplitter
import Network_TF


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dot_0_layer(input_layer, synaptic_weights):
    return layer_0_activator(np.dot(input_layer, synaptic_weights.T))


def layer_0_activator(weights_sum):
    # return max(0, weights_sum)
    result = [[0] * weights_sum[0]] * weights_sum
    for i in range(len(weights_sum)):
        sample_result = [0] * weights_sum[0]
        for k in range(len(weights_sum[i])):

            threshold = 1.8  # 1.79
            if weights_sum[i][k] >= threshold:
                sample_result[k] = weights_sum[i][k] - 2.2  # 0.79
            else:
                sample_result[k] = 0

        result[i] = sample_result
    result = np.array(result)
    # print(result)
    # exit(0)
    return np.array(result)


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class EngineeringProject(QtWidgets.QMainWindow):
    def __init__(self):
        super(EngineeringProject, self).__init__()
        uic.loadUi('gui_ep_ocr.ui', self)
        self.show()

        self.DEBUG = True
        self.font = cv2.FONT_HERSHEY_PLAIN

        self.btn_split.clicked.connect(self.split)
        self.btn_generate.clicked.connect(self.generate)
        self.btn_mlp_train_start.clicked.connect(self.mlp_train_start)
        self.btn_mlp_train_save.clicked.connect(self.mlp_train_save)
        self.btn_ocr_load_model.clicked.connect(self.ocr_load_model)
        self.btn_ocr_single.clicked.connect(self.ocr_single)
        self.btn_ocr_camera_start.clicked.connect(self.ocr_camera_start)
        self.btn_ocr_camera_pause.clicked.connect(self.ocr_camera_pause)
        self.btn_ocr_camera_stop.clicked.connect(self.ocr_camera_stop)

        self.slider_tolerance.valueChanged.connect(self.tolerance_changed)
        self.slider_height.valueChanged.connect(self.height_changed)
        self.slider_width.valueChanged.connect(self.width_changed)
        self.slider_min_area.valueChanged.connect(self.min_area_changed)
        self.slider_max_area.valueChanged.connect(self.max_area_changed)

        self.label_tolerance.setText(str(self.slider_tolerance.value()))
        self.label_height.setText(str(self.slider_height.value()))
        self.label_width.setText(str(self.slider_width.value()))
        self.label_min_area.setText(str(self.slider_min_area.value()))
        self.label_max_area.setText(str(self.slider_max_area.value()))

        self.camera_running = False
        self.camera_paused = False
        self.cv_cap = None
        self.network = None
        self.text_labels = []
        self.train_images = np.array([])
        self.train_labels = np.array([])

    def tolerance_changed(self):
        self.label_tolerance.setText(str(self.slider_tolerance.value()))

    def height_changed(self):
        self.label_height.setText(str(self.slider_height.value()))

    def width_changed(self):
        self.label_width.setText(str(self.slider_width.value()))

    def min_area_changed(self):
        self.label_min_area.setText(str(self.slider_min_area.value()))

    def max_area_changed(self):
        self.label_max_area.setText(str(self.slider_max_area.value()))

    def split(self):
        if self.DEBUG:
            print('Splitting images...')
        entries = Path(self.splitter_input_folder.text())
        entry_counter = 0
        for entry in entries.iterdir():
            splitter = ImageSplitter.ImageSplitter(cv2.imread(str(entry), 0),
                                                   min_area_threshold=self.slider_min_area.value(),
                                                   max_area=self.slider_max_area.value(),
                                                   tolerance_factor=self.slider_tolerance.value())

            for i in range(len(splitter.letters)):
                cv2.imwrite(self.splitter_save_folder.text() +
                            '/' + str(entry_counter) + '_' + str(i) + '.png', splitter.letters[i])
            entry_counter += 1
        if self.DEBUG:
            print('Splitting done.')

    def generate(self):
        if self.DEBUG:
            print('Generating dataset...')
        entries = Path(self.generator_image_folder.text())
        images = []
        classes = 0
        last_classes = []
        for entry in entries.iterdir():
            class_num = int(os.path.splitext(entry.name)[0].split('_')[1])
            if class_num > classes:
                classes = class_num
            if class_num not in last_classes:
                if not os.path.exists(self.generator_output.text() + '/' + str(class_num)):
                    os.mkdir(self.generator_output.text() + '/' + str(class_num))
                images.append([])
                last_classes.append(class_num)
        classes += 1

        for entry in entries.iterdir():
            class_num = int(os.path.splitext(entry.name)[0].split('_')[1])
            images[class_num].append(cv2.imread(str(entry), 0))

        for i in range(classes):
            generations_per_image = int(self.generator_images_n.value() / len(images[i]))
            generated_n = 0
            for image in images[i]:
                for k in range(generations_per_image):
                    temp_image = image.copy()
                    if self.generator_distortions.isChecked():

                        # Rotation
                        temp_image = 255 - temp_image
                        num_rows, num_cols = temp_image.shape[:2]
                        rotation_matrix = cv2.getRotationMatrix2D(
                            (num_cols / 2, num_rows / 2),
                            random.randrange(-7, 7), 1)
                        temp_image = cv2.warpAffine(temp_image, rotation_matrix, (num_cols, num_rows))

                        # Pixel shift
                        shift_size_x = random.randrange(-2, 2)
                        shift_size_y = random.randrange(-2, 2)
                        rows, cols = temp_image.shape
                        M = np.float32([[1, 0, shift_size_x], [0, 1, shift_size_y]])
                        temp_image = cv2.warpAffine(temp_image, M, (cols, rows))
                        temp_image = 255 - temp_image

                        # Brightness noise
                        temp_image = temp_image.astype('float32')
                        temp_image += random.randrange(-60, 60)
                        temp_image = np.clip(temp_image, 0, 255)
                        temp_image = temp_image.astype('uint8')

                        # Pixel noise
                        for y in range(image.shape[0]):
                            for x in range(image.shape[1]):
                                if random.randrange(0, 2) and temp_image[y][x] > 0:
                                    brightness_pixel = temp_image[y][x] + random.randrange(-20, 20)
                                    if brightness_pixel > 255: brightness_pixel = 255
                                    if brightness_pixel < 0: brightness_pixel = 0
                                    temp_image[y][x] = brightness_pixel

                    cv2.imwrite(self.generator_output.text() +
                                '/' + str(i) + '/' + str(generated_n) + '.png', temp_image)
                    generated_n += 1
        if self.DEBUG:
            print('Generating done.')

    def mlp_train_start(self):
        self.train_images = []
        self.train_labels = []
        with open(self.mlp_labels.text()) as f:
            self.text_labels = json.load(f)
        self.text_labels = np.asarray(self.text_labels)

        for i in range(len(self.text_labels)):
            entries = Path(self.mlp_train_folder.text() + '/' + str(i))
            for entry in entries.iterdir():
                self.train_images.append(1.0 - (cv2.imread(str(entry), 0).flatten() / 255.0))
                self.train_labels.append(i)

        combined_lists = list(zip(self.train_images, self.train_labels))
        random.shuffle(combined_lists)
        train_images, train_labels = zip(*combined_lists)

        self.network = Network_TF.Network(images=np.array(train_images), labels=np.array([train_labels]),
                                          train_output_labels=len(self.text_labels),
                                          train_iterations=self.mlp_train_iterations.value())
        self.network.start_training()

    def mlp_train_save(self):
        self.network.save_weights(self.mlp_train_save_file.text())

    def ocr_load_model(self):
        with open(self.mlp_labels.text()) as f:
            self.text_labels = np.asarray(json.load(f))
        self.network = Network_TF.Network()
        self.network.load_weights(self.ocr_model_file.text())

    def ocr_single(self):
        self.ocr_text.setPlainText(self.proceed_frame(cv2.imread(str(self.ocr_test_image.text()), 0)))

    def proceed_frame(self, image):
        splitter = ImageSplitter.ImageSplitter(image, min_area_threshold=self.slider_min_area.value(),
                                               max_area=self.slider_max_area.value(),
                                               tolerance_factor=self.slider_tolerance.value())
        debug_image = cv2.cvtColor(splitter.image.copy(), cv2.COLOR_GRAY2RGB)
        text_combined = ''

        if len(splitter.contours) > 0:
            self.network.images = []
            for letter in splitter.letters:
                self.network.images.append(1.0 - letter.flatten() / 255.0)
            self.network.images = np.array(self.network.images)

            predicted_classes = self.network.predict()

            letters = []
            for i in range(len(predicted_classes)):
                letters.append(self.text_labels[predicted_classes[i]])
                cv2.putText(debug_image, letters[i],  # letters[i], str(i)
                            (splitter.contours[i][0][0][0], splitter.contours[i][0][0][1]),
                            self.font, 1, (255, 0, 0), 1, cv2.LINE_AA)

            text_combined = ''
            lines = splitter.make_text(self.slider_height.value(), self.slider_width.value())

            for line in lines:
                for word in line:
                    for letter in word:
                        text_combined += letters[letter]
                        cv2.drawContours(debug_image, splitter.contours, letter, (0, 255, 0), 1)
                    text_combined += ' '
                text_combined += '\n'

        self.ocr_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))))

        return text_combined

    def ocr_camera_start(self):
        self.ocr_text.setPlainText('See console for output.')
        if not self.camera_paused:
            self.camera_running = True
            self.cv_cap = cv2.VideoCapture(self.ocr_camera_id.value(), cv2.CAP_DSHOW)
            thread = threading.Thread(target=self.camera_process)
            thread.start()
        self.camera_paused = False
        self.ocr_camera_id.setEnabled(False)
        self.btn_ocr_camera_start.setEnabled(False)
        self.btn_ocr_camera_pause.setEnabled(True)
        if self.DEBUG:
            print('Camera started.')

    def ocr_camera_pause(self):
        self.camera_paused = True
        self.btn_ocr_camera_start.setEnabled(True)
        self.btn_ocr_camera_pause.setEnabled(False)

    def camera_process(self):
        while self.camera_running:
            if not self.camera_paused:
                ret, img = self.cv_cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(self.proceed_frame(img))
                # self.ocr_text.setPlainText
        self.cv_cap.release()

    def ocr_camera_stop(self):
        self.camera_paused = False
        self.camera_running = False
        self.ocr_camera_id.setEnabled(True)
        self.btn_ocr_camera_start.setEnabled(True)
        self.btn_ocr_camera_pause.setEnabled(False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = EngineeringProject()
    window.show()
    app.exec_()
