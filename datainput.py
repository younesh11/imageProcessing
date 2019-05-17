import cv2
import os
import storage as st


faceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = os.getcwd()
print(path)

labelName = input("enter name: ")
tPath = path + "/" + "trainingData" 
mainPath = path + "/" + "trainingData" + "/" + labelName
print(mainPath)
os.makedirs(tPath + "/" + labelName)

cap =  cv2.VideoCapture(0)
count = 0

while True:

    ret, frame = cap.read()
    if st.faceExtractor(frame) is not None:
        count+=1
        face = cv2.resize(st.faceExtractor(frame),(100, 100))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        name = labelName + str(count) + '.jpg'
        print(name)

        cv2.imwrite(os.path.join(mainPath, name), face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('DataCollection', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Data Collection complete")



