'''
(c) 2018 Dzung Pham
This script converts the FER-2013 dataset from .csv to .npy
CSV format: emotion,pixels,Usage
'''

from config import *
import csv
import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier(CASCADE_FACE_PATH)

def _createLabelVec(label):
    vec = np.zeros(len(EMOTIONS))
    vec[int(label)] = 1.0
    return vec

def _createImage(data):
    image = np.fromstring(str(data), dtype = np.uint8, sep = ' ')
    image = image.reshape((IMG_SIZE, IMG_SIZE))
    face = face_cascade.detectMultiScale(image)
    if (len(face) == 0):
        return image
    (x, y, w, h) = face[0]
    img = image[y:(y+h)][x:(x+w)]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation = cv2.INTER_CUBIC)
    return img

if __name__ == '__main__':
    with open(CSV_PATH) as csvfile:
        reader = csv.reader(csvfile)
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        validation_images = []
        validation_labels = []
        count = 0
        next(reader, None) # Skip header
        
        print("Converting...")
        for row in reader:
            count += 1
            label = _createLabelVec(row[0])
            image = _createImage(row[1]).astype(np.float64)
            usage = row[2]
            
            if (usage == 'Training'):
                train_images.append(image)
                train_labels.append(label)
            elif (usage == 'PublicTest'):
                validation_images.append(image)
                validation_labels.append(label)
            else:
                test_images.append(image)
                test_labels.append(label)

            if (count % 1000 == 0):
                print('{} images converted'.format(count));

        np.save(TRAINING_DATA_PATH, train_images)
        np.save(TRAINING_LABEL_PATH, train_labels)
        np.save(VALIDATION_DATA_PATH, validation_images)
        np.save(VALIDATION_LABEL_PATH, validation_labels)        
        np.save(TEST_DATA_PATH, test_images)
        np.save(TEST_LABEL_PATH, test_labels)
        print('All images converted')
