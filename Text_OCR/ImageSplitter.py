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

import cv2
import numpy as np


class ImageSplitter:
    def __init__(self, image, binary_threshold=127,
                 splitted_width=20,
                 splitted_height=30,
                 min_area_threshold=4,
                 max_area=500,
                 tolerance_factor=61):

        self.image = image
        self.letters = []
        self.contours = []
        self.tolerance_factor = tolerance_factor
        ret, self.binarized = cv2.threshold(self.image, binary_threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(self.binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        avg_contour_area = 0
        for contour in contours:
            avg_contour_area += cv2.contourArea(contour)
        avg_contour_area /= len(contours)

        for contour in contours:
            if max_area > cv2.contourArea(contour) > avg_contour_area / min_area_threshold:
                [x, y, w, h] = cv2.boundingRect(contour)
                tl = [[x, y]]
                tr = [[x + w, y]]
                br = [[x + w, y + h]]
                bl = [[x, y + h]]
                self.contours.append(np.array([tl, tr, br, bl]))

        self.contours.sort(key=lambda sort_x: self.get_contour_precedence(sort_x, self.image.shape[1]))

        for contour in self.contours:
            letter = self.image[contour[0][0][1]: contour[2][0][1], contour[0][0][0]: contour[2][0][0]]
            letter = cv2.resize(letter, (splitted_width, splitted_height), interpolation=cv2.INTER_NEAREST)
            self.letters.append(letter)

    def get_contour_precedence(self, contour, cols):
        origin = cv2.boundingRect(contour)
        return ((origin[1] // self.tolerance_factor) * self.tolerance_factor) * cols + origin[0]

    def make_text(self, height_threshold, width_threshold, debug=False):

        last_contour_right_center = [0, 0]
        word = []
        words = []
        lines = []

        for i in range(len(self.contours)):
            if i == 0:
                word = [i]
                if debug:
                    print('New letter found: ', i)
                last_contour_right_center = self.contour_right_center(i)
            else:
                contour_left_center = self.contour_left_center(i)
                diff_x = abs(last_contour_right_center[0] - contour_left_center[0])
                diff_y = abs(last_contour_right_center[1] - contour_left_center[1])
                if debug:
                    print('Diff_x:', diff_x, ' Diff_y:', diff_y)

                if diff_y < height_threshold:  # In one line
                    if diff_x < width_threshold:  # In one word
                        word.append(i)
                        if debug:
                            print('New letter found: ', i)
                    else:
                        words.append(word)
                        word = [i]
                        if debug:
                            print('New word found: ', word)

                else:
                    words.append(word)
                    if debug:
                        print('New line found: ', words)
                    lines.append(words)
                    words = []
                    word = [i]
                    if debug:
                        print('New letter found: ', i)
                last_contour_right_center = self.contour_right_center(i)

        words.append(word)
        lines.append(words)
        if debug:
            print('Last line: ', words)

        return lines

    def contour_left_center(self, index):
        c_x = self.contours[index][0][0][0]
        c_y = int((self.contours[index][0][0][1] + self.contours[index][3][0][1]) / 2)
        return [c_x, c_y]

    def contour_right_center(self, index):
        c_x = self.contours[index][1][0][0]
        c_y = int((self.contours[index][0][0][1] + self.contours[index][3][0][1]) / 2)
        return [c_x, c_y]
