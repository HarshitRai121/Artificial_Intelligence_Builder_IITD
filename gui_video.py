import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = FacialExpressionModel("model_a1.json", "model_weights1.h5")
facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
cap = None

def start_video():
    global cap
    cap = cv2.VideoCapture(0)
    video_loop()

def stop_video():
    global cap
    if cap is not None:
        cap.release()
    cap = None
    video_label.configure(image='')
    app.update()

def video_loop():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = facec.detectMultiScale(gray_image, 1.3, 5)
            
            for (x, y, w, h) in faces:
                fc = gray_image[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
                emotion = EMOTIONS_LIST[np.argmax(pred)]
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        app.after(10, video_loop)

app = tk.Tk()
app.title("Real-time Emotion Detector")

start_button = Button(app, text="Start Video", width=15, command=start_video)
start_button.pack(padx=10, pady=5)

stop_button = Button(app, text="Stop Video", width=15, command=stop_video)
stop_button.pack(padx=10, pady=5)

video_label = Label(app)
video_label.pack()

app.mainloop()
