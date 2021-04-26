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

import random
import threading

import cv2
import numpy as np


# ------------------ USAGE ------------------ #
# sam_worker = SAM_worker.SAM(frame)
# sam_worker.split_and_merge()
# cv2.imshow('SAM', sam_worker.debug)
# ------------------------------------------- #
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))


class SAM:
    def __init__(self, image):
        self.image = image  # Input grayscale image
        self.im_height = self.image.shape[0]  # Height of input image
        self.im_width = self.image.shape[1]  # Width of input image
        self.regions = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        self.debug = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        self.checked_surface = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        # np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)

        self.threshold = 120
        self.checked_surface_mean = 0
        self.checked_surface_mean_last = 0
        self.contours = []

    def split_and_merge(self):
        self.regions = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        self.contours = []

        # Splitting
        print('Splitting ...')
        self.recursive_splitting([0, 0], [self.im_width - 1, 0],
                                 [self.im_width - 1, self.im_height - 1], [0, self.im_height - 1])
        self.contours = np.array(self.contours)
        print('Splitting done!')

        # Merging
        print('Merging ...')
        self.checked_surface = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        cv2.drawContours(self.regions, self.contours, -1, 255)

        # NUMBER OF THREADS
        for i in range(2):
            thread = threading.Thread(target=self.graphical_merge)
            thread.start()

        debug_counter = 0
        while np.mean(self.checked_surface) < 255:
            self.debug = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
            regions_rgb = cv2.merge(
                [np.zeros((self.im_height, self.im_width), dtype=np.uint8), self.regions,
                 np.zeros((self.im_height, self.im_width), dtype=np.uint8)])
            self.debug = cv2.add(self.debug, regions_rgb)

            # Show progress
            debug_counter += 1
            if debug_counter > 40:
                debug_counter = 0
                progress(self.checked_surface_mean, 255)
                # cv2.imshow('checked_surface', self.checked_surface)
                # cv2.imshow('debug', self.debug)
                # cv2.waitKey(1)

        print('Merging done!')

        # Finding contours
        # print('Finding contours ...')
        # self.found_contours()
        # print('Done! Threshold: ' + str(self.threshold) + ' Contours: ' + str(len(self.contours)))

        # self.contours, hierarchy = cv2.findContours(self.regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def recursive_splitting(self, tl, tr, br, bl):  # tl[0] = X (height), tl[1] = Y (width)
        segment = self.image[tl[1]: br[1], tl[0]: br[0]]

        if segment.max() - segment.min() > self.threshold:
            if segment.shape[0] >= 6 and segment.shape[1] >= 8:

                tl_tr_mid = [int((tr[0] + tl[0]) / 2), tl[1]]
                tr_br_mid = [tr[0], int((br[1] + tr[1]) / 2)]
                br_bl_mid = [int((br[0] + bl[0]) / 2), br[1]]
                bl_tl_mid = [bl[0], int((bl[1] + tl[1]) / 2)]
                all_mid = [tl_tr_mid[0], bl_tl_mid[1]]

                self.recursive_splitting(tl, tl_tr_mid, all_mid, bl_tl_mid)  # TL
                self.recursive_splitting(tl_tr_mid, tr, tr_br_mid, all_mid)  # TR
                self.recursive_splitting(all_mid, tr_br_mid, br, br_bl_mid)  # BR
                self.recursive_splitting(bl_tl_mid, all_mid, br_bl_mid, bl)  # BL
            else:
                self.contours.append([tl, tr, br, bl])

        else:
            self.contours.append([tl, tr, br, bl])

    def graphical_merge(self):
        kernel_full = np.ones((3, 3), np.uint8)
        kernel_big = np.ones((5, 5), np.uint8)

        self.checked_surface = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        expansion_seed = [1, 1]
        self.checked_surface_mean_last = 0
        while True:
            filled = self.regions.copy()
            cv2.floodFill(filled, None, (expansion_seed[0], expansion_seed[1]), 127)
            fill_mask = cv2.inRange(filled, 127, 127)
            region_mask = cv2.dilate(fill_mask, kernel_full)
            combined_region_mask = region_mask.copy()

            contours, hierarchy = cv2.findContours(cv2.dilate(fill_mask, kernel_big), cv2.RETR_LIST,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            contours = np.array(contours[0])
            horizontal_expansion = []
            vertical_expansion = []

            for i in range(len(contours)):
                dot_1_x = contours[i][0][0]
                dot_2_x = contours[i + 1][0][0] if i < len(contours) - 1 else contours[0][0][0]
                dot_1_y = contours[i][0][1]
                dot_2_y = contours[i + 1][0][1] if i < len(contours) - 1 else contours[0][0][1]
                if dot_1_x == dot_2_x and dot_1_x != self.image.shape[1] - 1:
                    # Vertical
                    for y in range(min(dot_1_y, dot_2_y), max(dot_1_y, dot_2_y)):
                        if self.regions[y, dot_1_x] == 255:
                            vertical_expansion.append([dot_1_x, y])
                else:
                    # Horizontal
                    for x in range(min(dot_1_x, dot_2_x), max(dot_1_x, dot_2_x)):
                        if self.regions[dot_1_y, x] == 255:
                            horizontal_expansion.append([x, dot_1_y])

            last_y = -1
            for i in range(len(horizontal_expansion)):
                if horizontal_expansion[i][1] <= 0 or horizontal_expansion[i][1] >= self.image.shape[0] - 1:
                    continue
                elif horizontal_expansion[i][1] != last_y:
                    last_y = horizontal_expansion[i][1]
                else:
                    diff_x = int((horizontal_expansion[i - 1][0] + horizontal_expansion[i][0]) / 2)

                    # Create new region around horizonal lines
                    new_region = self.regions.copy()
                    cv2.floodFill(new_region, None, (diff_x, last_y), 127)
                    new_region = cv2.dilate(cv2.inRange(new_region, 127, 127), kernel_full)
                    region_max = cv2.bitwise_and(region_mask, self.image).max()
                    region_min = cv2.bitwise_or(cv2.bitwise_and(region_mask, self.image),
                                                cv2.bitwise_not(region_mask)).min()
                    new_region_max = cv2.bitwise_and(new_region, self.image).max()
                    new_region_min = cv2.bitwise_or(cv2.bitwise_and(new_region, self.image),
                                                    cv2.bitwise_not(new_region)).min()

                    if (new_region_max - region_min <= self.threshold
                            and region_max - new_region_min <= self.threshold):
                        combined_region_mask = cv2.bitwise_or(combined_region_mask, new_region)

            last_x = -1
            for i in range(len(vertical_expansion)):
                if vertical_expansion[i][0] <= 0 or vertical_expansion[i][0] >= self.image.shape[1] - 1:
                    continue
                elif vertical_expansion[i][0] != last_x:
                    last_x = vertical_expansion[i][0]
                else:
                    diff_y = int((vertical_expansion[i - 1][1] + vertical_expansion[i][1]) / 2)

                    # Create new region around vertical lines
                    new_region = self.regions.copy()
                    cv2.floodFill(new_region, None, (last_x, diff_y), 127)
                    new_region = cv2.dilate(cv2.inRange(new_region, 127, 127), kernel_full)
                    region_max = cv2.bitwise_and(region_mask, self.image).max()
                    region_min = cv2.bitwise_or(cv2.bitwise_and(region_mask, self.image),
                                                cv2.bitwise_not(region_mask)).min()
                    new_region_max = cv2.bitwise_and(new_region, self.image).max()
                    new_region_min = cv2.bitwise_or(cv2.bitwise_and(new_region, self.image),
                                                    cv2.bitwise_not(new_region)).min()

                    if (new_region_max - region_min <= self.threshold
                            and region_max - new_region_min <= self.threshold):
                        combined_region_mask = cv2.bitwise_or(combined_region_mask, new_region)

                    """
                    if (new_region_max - region_min <= self.merge_threshold
                        and region_max - new_region_min <= self.merge_threshold) \
                            or (np.mean(new_region) < 0.03 or np.mean(region_mask) < 0.03):
                    """

            self.checked_surface = cv2.bitwise_or(self.checked_surface, combined_region_mask)

            combined_region_mask = cv2.erode(combined_region_mask, kernel_full)

            self.regions = cv2.subtract(self.regions, combined_region_mask)

            self.checked_surface_mean = np.mean(self.checked_surface)
            if self.checked_surface_mean == 255:
                break
            elif self.checked_surface_mean_last != self.checked_surface_mean:
                self.checked_surface_mean_last = self.checked_surface_mean
                expansion_seed_founded = True
            else:
                expansion_seed_founded = False

            while not expansion_seed_founded:
                expansion_seed = [random.randrange(self.image.shape[1]), random.randrange(self.image.shape[0])]
                if not self.checked_surface[expansion_seed[1], expansion_seed[0]] \
                        and not self.regions[expansion_seed[1], expansion_seed[0]]:
                    expansion_seed_founded = True
            """
            for ex_y in range(self.image.shape[0]):
                if expansion_seed_founded:
                    break
                for ex_x in range(self.image.shape[1]):
                    if not self.checked_surface[ex_y, ex_x] and not self.regions[ex_y, ex_x]:
                        expansion_seed = [ex_x, ex_y]
                        expansion_seed_founded = True
                        break
            """

            # print('expansion_seed: ' + str(expansion_seed))
            # progress(self.checked_surface_mean, 255)

            # self.debug = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
            # regions_rgb = cv2.merge(
            #     [np.zeros((self.im_height, self.im_width), dtype=np.uint8), self.regions,
            #      np.zeros((self.im_height, self.im_width), dtype=np.uint8)])
            # self.debug = cv2.add(self.debug, regions_rgb)
            # cv2.imshow('self.checked_surface', self.checked_surface)
            # cv2.imshow('debug', self.debug)
            # cv2.waitKey(1)

        cv2.destroyAllWindows()

    def found_contours(self):
        self.contours = []
        kernel_full = np.ones((3, 3), np.uint8)

        expansion_seed = [1, 1]
        while True:
            filled = self.regions.copy()
            cv2.floodFill(filled, None, (expansion_seed[0], expansion_seed[1]), 127)
            fill_mask = cv2.inRange(filled, 127, 127)
            region_mask = cv2.dilate(fill_mask, kernel_full)

            contours, hierarchy = cv2.findContours(region_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            self.contours.append(np.array(contours[0]))

            self.checked_surface = cv2.bitwise_or(self.checked_surface, region_mask)

            self.checked_surface_mean = np.mean(self.checked_surface)
            if self.checked_surface_mean == 255:
                break

            expansion_seed_founded = False

            while not expansion_seed_founded:
                expansion_seed = [random.randrange(self.image.shape[1]), random.randrange(self.image.shape[0])]
                if not self.checked_surface[expansion_seed[1], expansion_seed[0]] \
                        and not self.regions[expansion_seed[1], expansion_seed[0]]:
                    expansion_seed_founded = True
            """
            for ex_y in range(self.image.shape[0]):
                if expansion_seed_founded:
                    break
                for ex_x in range(self.image.shape[1]):
                    if not self.checked_surface[ex_y, ex_x] and not self.regions[ex_y, ex_x]:
                        expansion_seed = [ex_x, ex_y]
                        expansion_seed_founded = True
                        break
            """
            # print('expansion_seed: ' + str(expansion_seed))
            # progress(self.checked_surface_mean, 255)

            cv2.imshow('self.checked_surface', self.checked_surface)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
