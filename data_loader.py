'''
(c) 2018 Dzung Pham
This script provides a class that loads the dataset
'''

from config import *
import numpy as np

class DataLoader:
    def __init__(self):
        pass

    def load_data(self):
        # Load data
        self._train_images = np.load(TRAINING_DATA_PATH)
        self._train_labels = np.load(TRAINING_LABEL_PATH)
        self._validation_images = np.load(VALIDATION_DATA_PATH)
        self._validation_labels = np.load(VALIDATION_LABEL_PATH)
        self._test_images = np.load(TEST_DATA_PATH)
        self._test_labels = np.load(TEST_LABEL_PATH)

        # Reshape images to 4D for tensors and labels to 2D
        self._train_images = self._train_images.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
        self._validation_images = self._validation_images.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
        self._test_images = self._test_images.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
        
        self._train_labels = self._train_labels.reshape([-1, len(EMOTIONS)])
        self._validation_labels = self._validation_labels.reshape([-1, len(EMOTIONS)])
        self._test_labels = self._test_labels.reshape([-1, len(EMOTIONS)])

    @property
    def train_images(self):
        return self._train_images

    @property
    def validation_images(self):
        return self._validation_images        
    
    @property
    def test_images(self):
        return self._test_images

    @property
    def train_labels(self):
        return self._train_labels    

    @property
    def validation_labels(self):
        return self._validation_labels
        
    @property
    def test_labels(self):
        return self._test_labels
