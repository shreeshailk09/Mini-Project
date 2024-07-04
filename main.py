import cv2
import numpy as np


face_prototxt_path = 'deploy.prototxt' 
face_caffemodel_path = 'res10_300x300_ssd_iter_140000.caffemodel'
gender_prototxt_path = 'gender_deploy.prototxt'
gender_caffemodel_path = 'gender_net.caffemodel'
age_prototxt_path = 'age_deploy.prototxt'
age_caffemodel_path = 'age_net.caffemodel'


face_net = cv2.dnn.readNetFromCaffe(face_prototxt_path, face_caffemodel_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_caffemodel_path)
age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_caffemodel_path)


gender_labels = ['Male', 'Female']

age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-18)', '(19-26)', '(26-30)', '(38-43)', '(48-53)', '(60-100)']


cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

    if not ret:
        break

   
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

   
    face_net.setInput(blob)

    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  
        if confidence > 0.5: 
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)

            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(frame.shape[1], endX)
            endY = min(frame.shape[0], endY)

            face_roi = frame[startY:endY, startX:endX]

            if face_roi.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            gender_net.setInput(blob)
            age_net.setInput(blob)

            gender_preds = gender_net.forward()
            age_preds = age_net.forward()

            gender = gender_labels[gender_preds[0].argmax()]
            age_index = age_preds[0].argmax()
            age = age_ranges[age_index]

            label = f'{gender}, {age}'

           
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
           
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
