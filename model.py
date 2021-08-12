'''
Camera Classifier v0.1 Alpha
Copyright (c) NeuralNine

Instagram: @neuralnine
YouTube: NeuralNine
Website: www.neuralnine.com
'''

from os import terminal_size
from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL
import glob
import pickle
import os

class Model:

    def __init__(self):
        # Load from file
        self.model_file_name = 'trained_model.pkl'
        if os.path.isfile(self.model_file_name):
            with open(self.model_file_name, 'rb') as file:
                self.model = pickle.load(file)
        else:
            self.model = LinearSVC()

        self.shape = 16950 #previous 16800

    def train_model(self, counters):
        img_list = np.array([])
        class_list = np.array([])
        samples_size = [len(glob.glob1("./1","*.jpg")), len(glob.glob1("./2","*.jpg")), len(glob.glob1("./3","*.jpg"))]

        #for i in range(1, counters[0]):
        for i in range(0, samples_size[0]):
            img = cv.imread(f'1/frame{i + 1}.jpg')[:, :, 0]
            img = img.reshape(self.shape)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        #for i in range(1, counters[1]):
        for i in range(0, samples_size[1]):
            img = cv.imread(f'2/frame{i + 1}.jpg')[:, :, 0]
            img = img.reshape(self.shape)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        #for i in range(1, counters[2]):
        for i in range(0, samples_size[2]):
            img = cv.imread(f'3/frame{i + 1}.jpg')[:, :, 0]
            img = img.reshape(self.shape)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(samples_size[0] + samples_size[1] + samples_size[2], self.shape)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")
        
        with open(self.model_file_name, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')[:, :, 0]
        img = img.reshape(self.shape)
        prediction = self.model.predict([img])

        return prediction[0]