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
import sys
import threading
import tensorflow as tf

import numpy as np


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
        self.model = None

    def save_weights(self, file):
        if self.debug:
            print('Saving model...')
        self.model.save(os.path.splitext(file)[0] + '.h5')
        if self.debug:
            print('Done. File', file, 'saved.')

    def load_weights(self, file):
        if self.debug:
            print('Loading model...')
        self.model = tf.keras.models.load_model(os.path.splitext(file)[0] + '.h5')
        if self.debug:
            print('Done. Model from file', file, 'loaded.')

    def start_training(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(600,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.train_output_labels)
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        # TRAINING
        thread = threading.Thread(target=self.training)
        thread.start()

    def predict(self):
        predicted_classes = []
        predictions = self.model.predict(self.images)
        for prediction in predictions:
            predicted_classes.append(np.argmax(prediction))
        return predicted_classes

    def training(self):
        # noinspection PyBroadException
        try:
            self.labels = np.reshape(self.labels, len(self.labels[0]))
            self.model.fit(self.images, self.labels, epochs=self.train_iterations)
        except:
            print(sys.exc_info())
