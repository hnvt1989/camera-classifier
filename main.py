'''
Camera Classifier v0.1 Alpha
Copyright (c) NeuralNine

Instagram: @neuralnine
YouTube: NeuralNine
Website: www.neuralnine.com
'''
import app
import os

def main():
    app.App(window_title="Camera Classifier v0.1 Alpha")

if __name__ == "__main__":
    if os.path.isfile('trained_model.pkl'):
        print('file exist')
        
    main()