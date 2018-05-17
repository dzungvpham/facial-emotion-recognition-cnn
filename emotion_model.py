'''
(c) 2018 Dzung Pham
This script provides a class that trains a Convolutional Neural Networks model
for facial emotion recognition.

Current model spec (trained on all 28k images and tested on ~3.5k images after 80 epochs):
Validation Acc: 64.25%
Test Acc: 65.33%

TODO: Implement better arg parse
'''

import sys
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.optimizers import Momentum
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import DataPreprocessing
from config import *
from data_loader import DataLoader

class EmotionModel:
    __slots__ = ['dataset', 'network', 'model']

    def __init__(self):
        self.dataset = DataLoader()
        self.network = None
        self.model = None

    def build_network(self):
        '''
        This method builds the network architecture for the CNN model.
        Modify as needed.
        '''
        
        # Image Preprocessing
        img_preprocess = DataPreprocessing()
        img_preprocess.add_samplewise_zero_center()
        img_preprocess.add_featurewise_stdnorm()

        # Image Augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle = 5.0)        

        # Input Layer
        self.network = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1],
                                  data_preprocessing = img_preprocess,
                                  data_augmentation = img_aug)
                                  
        # Convolution Layer 1
        self.network = conv_2d(self.network, 64, 5, activation = 'relu')
        self.network = batch_normalization(self.network)
        self.network = max_pool_2d(self.network, 3, strides = 2)

        # Convolution Layer 2
        self.network = conv_2d(self.network, 64, 5, activation = 'relu')
        self.network = batch_normalization(self.network)
        self.network = max_pool_2d(self.network, 3, strides = 2)

        # Convolution Layer 3
        self.network = conv_2d(self.network, 128, 4, activation = 'relu')
        self.network = batch_normalization(self.network)
        self.network = dropout(self.network, 0.2)

        # Penultimate FC Layer
        self.network = fully_connected(self.network, 3072, activation = 'relu')
        self.network = batch_normalization(self.network)

        # Final FC Layer
        self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
        
        # Create network
        optimizer = Momentum(learning_rate=0.01, lr_decay=0.99, decay_step=250) # Learning function
        self.network = regression(self.network,
                                  optimizer = optimizer,
                                  loss = 'categorical_crossentropy')

        # Create model
        self.model = tflearn.DNN(
            self.network,
            tensorboard_dir = TENSORBOARD_PATH,
            checkpoint_path = CHECKPOINT_PATH,
            max_checkpoints = 1,
            tensorboard_verbose = 1
        )

    def load_data(self):
        self.dataset.load_data()

    def start_training(self, model_path = None):
        self.load_data()

        if (self.network == None):
            self.build_network()
        if (model_path):
            self.load_model(model_path)

        print('Training initilized...')
        self.model.fit(
            self.dataset.train_images, self.dataset.train_labels,
            validation_set = (self.dataset.validation_images, self.dataset.validation_labels),
            n_epoch = 80,
            batch_size = 50,
            shuffle = True,
            show_metric = True,
            snapshot_epoch = True,
            run_id = 'emotion_recognition'
        )
        print('Training completed')

    def predict(self, image):
        image = image.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
        return self.model.predict(image)

    def save_model(self, path = MODEL_PATH):
        try:
            self.model.save(path)
            print('Model saved')
        except:
            raise Exception('Error saving model')

    def load_model(self, path = MODEL_PATH):
        try:
            self.model.load(path)
            print('Model loaded')
        except:
            raise Exception('Error loading model')

    def evaluate_model(self):
        ''' This method evaluates the model on the test set '''
        self.build_network()
        self.load_model()
        self.load_data()
        return self.model.evaluate(self.dataset.test_images,
                                    self.dataset.test_labels)

    def create_confusion_matrix(self):
        ''' This method creates a confusion matrix for the test set '''
        self.build_network()
        self.load_model()
        self.load_data()
        confusion_matrix = [[0 for _ in range(len(EMOTIONS))] for _ in range(len(EMOTIONS))]
        for img, label in zip(self.dataset.test_images,
                                self.dataset.test_labels):
            pred = np.argmax(self.predict(img))
            actual = np.argmax(label)
            confusion_matrix[pred][actual] += 1
        return confusion_matrix

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        model = EmotionModel()
        if len(sys.argv) == 1:
            model.start_training()
            model.save_model()
        else:
            model.start_training(sys.argv[1])
            model.save_model(sys.argv[1])
    else:
        print('Usage: python emotion_model.py <load_path>')
    
    
