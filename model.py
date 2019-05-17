import cv2
import os
import numpy as np
import storage as st


faces,faceID = st.labels_for_training_data('trainingData')
face_recognizer = st.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yaml')