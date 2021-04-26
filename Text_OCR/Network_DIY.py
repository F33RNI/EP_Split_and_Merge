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

import pickle
import random
import sys
import threading

import numpy as np


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


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))


class Network:
    def __init__(self, synaptic_weights_0=np.array([]), synaptic_weights_1=np.array([]),
                 images=np.array([]), labels=np.array([]),
                 train_iterations=20, train_output_labels=4, debug=True):
        self.synaptic_weights_0 = synaptic_weights_0
        self.synaptic_weights_1 = synaptic_weights_1
        self.images = images
        self.labels = labels
        self.train_iterations = train_iterations
        self.debug = debug
        self.train_output_labels = train_output_labels

    def save_weights(self, file):
        if self.debug:
            print('Saving weights...')
        compressed_data = [self.synaptic_weights_0, self.synaptic_weights_1]
        with open(file, 'wb') as filehandle:
            pickle.dump(compressed_data, filehandle)
        if self.debug:
            print('Done. File', file, 'saved.')

    def load_weights(self, file):
        if self.debug:
            print('Loading weights...')
        with open(file, 'rb') as filehandle:
            compressed_data = pickle.load(filehandle)
            self.synaptic_weights_0 = np.array(compressed_data[0])
            self.synaptic_weights_1 = np.array(compressed_data[1])
        if self.debug:
            print('Done. Weights from file', file, 'loaded.')

    def start_training(self):
        if len(self.synaptic_weights_0) == 0 or len(self.synaptic_weights_1) == 0:
            if self.debug:
                print('Weights are empty! Generating...')
            # Genarate weights
            images_flatten_size = len(self.images[0])
            self.synaptic_weights_0 = []
            for i in range(512):  # 4096
                string_array = [int(random.randrange(-1, 2)) for _ in range(3)] \
                               + [0 for _ in range(images_flatten_size - 3)]
                random.shuffle(string_array)
                self.synaptic_weights_0.append(string_array)
            self.synaptic_weights_0 = np.array(self.synaptic_weights_0)
            self.synaptic_weights_1 = np.array(2 * np.random.random((self.train_output_labels, 512)) - 1)  # 4096

        if self.debug:
            print('-------------------- TRAIN DATA --------------------')
            print('Shape of images: ' + str(self.images.shape))
            print('Shape of labels: ' + str(self.labels.shape))
            print('Arrays:')
            print(self.images)
            print()
            print(self.labels)
            print('--------------------- WEIGHTS ----------------------')
            print('Shape of synaptic_weights_0: ' + str(self.synaptic_weights_0.shape))
            print('Shape of synaptic_weights_1: ' + str(self.synaptic_weights_1.shape))
            print('Arrays:')
            print(self.synaptic_weights_0)
            print()
            print(self.synaptic_weights_1)
            print('----------------------------------------------------')

        # TRAINING
        thread = threading.Thread(target=self.training)
        thread.start()

    def predict(self):
        output_l0 = self.images
        output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
        output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))
        predicted_classes = []
        for output in output_l2:
            predicted_classes.append(np.argmax(output))
        return predicted_classes

    def training(self):
        # noinspection PyBroadException
        try:
            i = 0
            while i < self.train_iterations:
                output_l0 = self.images

                output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
                output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))

                # Layer 2 error calculations
                error_l2 = []
                for k in range(len(output_l2)):
                    a = []
                    for m in range(self.train_output_labels):
                        if m == self.labels[0][k]:
                            a.append(1 - output_l2[k][m])
                        else:
                            a.append(0 - output_l2[k][m])
                    error_l2.append(a)
                error_l2 = np.array(error_l2)

                adjustments_l2 = output_l1.T.dot(error_l2 * (output_l2 * (1 - output_l2)))
                self.synaptic_weights_1 += adjustments_l2.T

                # accuracy calculations
                predicted = []
                accuracy = 0
                for k in range(len(output_l2)):
                    predicted.append(np.argmax(output_l2[k]))
                    if np.argmax(output_l2[k]) == self.labels[0][k]:
                        accuracy += 1
                accuracy /= len(output_l2)
                predicted = np.array(predicted)

                if i % 1 == 0 and self.debug:
                    print('iteration:', i)
                    print('predicted:', predicted)
                    print('accuracy:', accuracy)
                    print()
                i += 1
                progress(i, self.train_iterations, 'Training...')

            if self.debug:
                print('Training done. Don\'t forget to save the weights!')

        except:
            print(sys.exc_info())
