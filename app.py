from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it does not exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your face, gender, and age detection models
face_prototxt_path = 'models/deploy.prototxt'
face_caffemodel_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
gender_prototxt_path = 'models/gender_deploy.prototxt'
gender_caffemodel_path = 'models/gender_net.caffemodel'
age_prototxt_path = 'models/age_deploy.prototxt'
age_caffemodel_path = 'models/age_net.caffemodel'

face_net = cv2.dnn.readNetFromCaffe(face_prototxt_path, face_caffemodel_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_caffemodel_path)
age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_caffemodel_path)

gender_labels = ['Male', 'Female']
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-18)', '(19-26)','(26-30)', '(35-53)', '(60-100)']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype(int)
                face_roi = image[startY:endY, startX:endX]

                if face_roi.size > 0:
                    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    gender_net.setInput(blob)
                    age_net.setInput(blob)

                    gender_preds = gender_net.forward()
                    age_preds = age_net.forward()

                    gender = gender_labels[gender_preds[0].argmax()]
                    age_index = age_preds[0].argmax()
                    age = age_ranges[age_index]

                    label = f'{gender}, {age}'
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', image)
        image_encoded = buffer.tobytes()
        return Response(image_encoded, mimetype='image/jpeg')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype(int)
                face_roi = frame[startY:endY, startX:endX]

                if face_roi.size > 0:
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
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
