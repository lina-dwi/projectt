import cv2
import numpy as np
import yaml
import os

# Baca data konfigurasi dari file YAML
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Baca nama kelas objek dari file YAML
classFile = 'coco.names'
classNames = []
with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Tambahkan kelas objek dari gambar-gambar di folder 'images'
image_folder = 'valid/images'
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]
for image_file in image_files:
    class_name = os.path.splitext(os.path.basename(image_file))[0]
    if class_name not in classNames:
        classNames.append(class_name)

# Konfigurasi model deteksi objek YOLO
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Inisialisasi model deteksi objek
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Konfigurasi kamera dan deteksi objek
cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Fungsi untuk mendeteksi objek dan menampilkan hasilnya
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w/2), int((det[1] * hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 0, 225), 2)

# Loop utama untuk mendeteksi objek menggunakan kamera real-time
while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
