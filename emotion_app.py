'''
(c) 2018 Dzung Pham
Facial Emotion Recognition App Using Convolution Neural Networks
Final Project for CS 373: Artificial Intelligence - Spring 2018, Williams College

Usage:
- Real-time: python emotion_app.py
    + Uses your webcam
- Video, Image, Directory: python emotion_app.py <path>
    + Video:        Classify each frame of the video and show it without saving
    + Image:        Classify the image, save it, and open a screen to show it
    + Directory:    Classify all images in the directory,
                    and save the predictions in the same directory

For each image/frame, the app finds all the faces in the image/frame using the
Haar Cascade classifier from OpenCV, crops the faces out, greyscales, resizes,
and passes them into the trained CNN model for prediction. Then, it draws a blue boundary
box around the face and writes the predicted emotion with probability on top of the box.

TODO: Implement saving video/webcam, and better arg parse
'''

import sys, os
import cv2
import numpy as np
from config import *
from emotion_model import *

# Supported image and video extensions for this app. OpenCV supports more than this.
IMG_EXT = ['.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.bmp', '.tiff', '.tif']
VIDEO_EXT = ['.avi', '.mp4']
DRAW_COLOR = (0, 255, 0) # Green

class EmotionApp:
    __slots__ = ['_face_cascade', '_model']

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(CASCADE_FACE_PATH)
        self._model = EmotionModel()
        self._model.build_network()
        self._model.load_model()

    def _classify(self, frame):
        prediction = self._model.predict(frame)
        if prediction is not None:
            classification = np.argmax(prediction)
            percentage = prediction[0][classification]/np.sum(prediction[0])
            return (EMOTIONS[classification], percentage)
        return('N/A', '-')

    def _classify_frame(self, frame):
        # Find all faces in frame
        faces = self._face_cascade.detectMultiScale(frame, 1.1, 10)
        if len(faces) == 0:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale
        for (x, y, w, h) in faces:
            # Format the face
            face = gray[y:(y + h), x:(x + w)]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE),
                             interpolation = cv2.INTER_CUBIC)
            face = face.astype(np.float64)

            # Pass into model and print out prediction & probability
            emotion, prob = self._classify(face)
            prob = round(prob * 100, 2)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                    color = DRAW_COLOR, thickness = 2)
            cv2.putText(frame, '{}: {}%'.format(emotion, prob), (x, y - 5),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = w/200,
                        color = DRAW_COLOR, thickness = 1, lineType = cv2.LINE_AA)

        return frame

    def predict_image(self, path, show = False):
        ''' Predicts an image and save the predicted image in the same directory
        '''
        file, ext = os.path.splitext(path)
        image = cv2.imread(path)
        img_predict = self._classify_frame(image)
        cv2.imwrite("{}-predict{}".format(file, ext), image)
        if show:
            cv2.imshow('Emotion Recognition App', img_predict)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def predict_video(self, path):
        ''' Predicts all frames in a video and replay it without saving
        Needs updating
        '''
        cap = cv2.VideoCapture(path)
        replay = []
        count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._classify_frame(frame)
            frame = cv2.putText(frame, 'Press Q to quit', (5, 30),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1,
                        color = DRAW_COLOR, thickness = 1, lineType = cv2.LINE_AA)
            replay.append(frame)

        cap.release()
        for frame in replay:
            cv2.imshow('Emotion Recognition App', frame)
            key_press = cv2.waitKey(50)
            if key_press and key_press == ord('q'):
                break
        cv2.destroyAllWindows()

    def predict_dir(self, path):
        ''' Predicts all images in a directory
        '''
        print('Predicting...')
        for file_name in os.listdir(path):
            _, ext = os.path.splitext(file_name)
            if (ext.lower() in IMG_EXT):
                self.predict_image(os.path.join(path, file_name))
        print('Prediction(s) completed')

    def predict_real_time(self):
        ''' Real-time classification using webcam.
        '''
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        while (True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # Flip horizontally to make it mirror-like
            frame = self._classify_frame(frame)
            frame = cv2.putText(frame, 'Press Q to quit', (5, 30),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1,
                        color = DRAW_COLOR, thickness = 1, lineType = cv2.LINE_AA)
            cv2.imshow('Emotion Recognition App', frame)
            key_press = cv2.waitKey(1)
            if key_press and key_press == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def parse_arg(self, path):
        ''' Parses the input path and chooses the appropriate action.
        '''
        _, ext = os.path.splitext(path) # Get the file extension
        if os.path.isfile(path):
            if (ext.lower() in IMG_EXT):
                self.predict_image(path, show = True)
            elif (ext.lower() in VIDEO_EXT): # Not working as expected
                self.predict_video(path)
            else:
                print('Invalid file!')
        elif os.path.isdir(path):
            self.predict_dir(path)
        else:
            print('Invalid path!')

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        app = EmotionApp()
        if len(sys.argv) == 1:
            print('Real-time Emotion Recognition')
            print('To quit, press Q')
            app.predict_real_time()
        else:
            app.parse_arg(sys.argv[1])
    else:
        print('Emotion Recognition App. 4 modes: real-time, video, image, directory')
        print('Usage: python emotion_app.py <path>')
        print('For real-time recognition: python emotion_app.py')
