import os
import cv2
import numpy as np
import storage as st


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yaml')
name = {0 : "younesh", 1 : "Saajan Danuwar"}
cap = cv2.VideoCapture(0)

while True:

    ret, test_img = cap.read()
    faces_detected, gray_img = st.faceDetection(test_img)
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+w, x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print("confidence:",confidence)
        print("label:",label)
        if label == 0:
            print('younesh')
        if label == 1:
            print('Saajan Danuwar')

        if label == 2:
            print('Rachita')

        st.draw_rect(test_img, face)
        predicted_name = name[label]
        print(predicted_name)

        confidence = int(100 * (1 - (confidence) / 300))
        print('confidence is', confidence)
        b = str(confidence)
        if confidence < 70:
            
            cv2.putText(test_img, b, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), lineType = cv2.LINE_AA)
        #cv2.putText(test_img,predicted_name , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), lineType = cv2.LINE_AA) 
        else:

            print(name[label])
            cv2.putText(test_img,predicted_name , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), lineType = cv2.LINE_AA) 
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition',resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
