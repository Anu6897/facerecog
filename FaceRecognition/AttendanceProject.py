import cv2
import numpy as np
import face_recognition
import os
import datetime
import time


attendance_list = {"ANURAAG ARAVINDAN": "Absent", "BILL GATES": "Absent", "ELON MUSK": "Absent", "JACK MA": "Absent"}
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            end = now.replace(hour=12, minute=45, second=0, microsecond=0)
            here = (now < end)
            dtString = now.strftime('%H:%M')
            f.writelines(f'\n{name},{dtString},{here}')


encodeListKnow = findEncodings(images)

t0 = time.time()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    cv2.imshow('Webcam', img)
    t1 = time.time()
    num_seconds = t1 - t0
    if num_seconds > 15:
        cv2.destroyAllWindows()
        cap.release()
        break

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            attendance_list[name] = "Present"

        else:
            name = 'Unknown'
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6),cv2.QT_FONT_NORMAL,0.75, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

present = 0
absent = 0
now = datetime.datetime.now()
dtString = now.strftime('%H:%M')
print("Attendance as per", dtString)
print("Name  Absent/Present")
for i in attendance_list:
    print(i, ":", attendance_list[i])
    if attendance_list[i] == "Present":
        present += 1
    elif attendance_list[i] == "Absent":
        absent += 1
print("Total Present :", present)
print("Total Absent :", absent)
attendance_list['Total Present'] = present
attendance_list['Total Absent'] = absent




       # k = cv2.waitKey(30) & 0xff

        #cv2.imshow('Webcam', img)
        #if k == 27:
         #   cv2.destroyAllWindows()
          #  cap.release()
           # break



# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
