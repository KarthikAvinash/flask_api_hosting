import cv2
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
import json

# Specify the file path
file_path = "labels.json"
with open(file_path, 'r') as json_file:
    labels_encoded = json.load(json_file)
print(labels_encoded)

def load_tflite_model(file):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    return interpreter

facenet_tfl_file = "facenet.tflite"
facenet_model = load_tflite_model(facenet_tfl_file)
facenet_input_details = facenet_model.get_input_details()
facenet_output_details = facenet_model.get_output_details()

cnn_tfl_file = "cnn_model.tflite"
cnn_model = load_tflite_model(cnn_tfl_file)
cnn_input_details = cnn_model.get_input_details()
cnn_output_details = cnn_model.get_output_details()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Load models
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    # ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    ret = ret.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def extract_embeddings(face_images):
    # Test the model on input data.
    input_shape = facenet_input_details[0]['shape']
    embeddings = []
    for face_image in face_images:
        face_image = pre_process(face_image)
        input_data = face_image.reshape(input_shape)
        facenet_model.set_tensor(facenet_input_details[0]['index'], input_data)
        facenet_model.invoke()
        output_data = facenet_model.get_tensor(facenet_output_details[0]['index'])
        print("Output data shape: ", len(output_data))
        print("tensor: ",output_data)
        for i in output_data:
            embeddings.append(i)
    return embeddings

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    return faces

def predict(image):
    # Predict labels for the image
    faces = detect_faces(image)
    print("total faces identified: ", len(faces))
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    # embeddings = extract_embeddings(face_images)
    embeddings = extract_embeddings(face_images)
    predicted_labels = []
    for embedding in embeddings:
        # print(embedding)
        # print(type(embedding))
        embedding = embedding.reshape(1, 512, 1).astype(np.float32)
        cnn_model.set_tensor(cnn_input_details[0]['index'], embedding)
        cnn_model.invoke()
        output_data = cnn_model.get_tensor(cnn_output_details[0]['index'])
        predicted_labels.append(output_data)
    return predicted_labels

# Read image file from request
image_file = r"C:\Users\Karthik Avinash\OneDrive\Desktop\6th Sem\Mini-project\0. Dataset\19_ipcv_dataset_with_annotation\21bcs133_v5_f024.jpg"
image = cv2.imread(image_file)

# Predict labels for the image
prediction = predict(image)
# Format and return prediction
labels = []
for i in prediction:
    # print(i)
    predicted_label_index = np.argmax(i)
    print("predicted label index: ",predicted_label_index)
    if predicted_label_index <= 27 and i[0][predicted_label_index]>=0.95:
        predicted_label = labels_encoded[str(predicted_label_index)]
    else:
        predicted_label = "UNKNOWN"
    labels.append(predicted_label)
unique_labels = list(set(labels))

print(unique_labels)
