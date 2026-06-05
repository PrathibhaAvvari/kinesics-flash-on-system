import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# FIX 1: Load pre-trained weights (not save)
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# FIX 2: Use os.path for cross-platform paths
cur_path = os.path.dirname(os.path.abspath(__file__))

emoji_dist = {
    0: os.path.join(cur_path, "Dataset_emojis", "angry.png"),
    1: os.path.join(cur_path, "Dataset_emojis", "disgusted.png"),
    2: os.path.join(cur_path, "Dataset_emojis", "fearful.png"),
    3: os.path.join(cur_path, "Dataset_emojis", "happy.png"),
    4: os.path.join(cur_path, "Dataset_emojis", "neutral.png"),
    5: os.path.join(cur_path, "Dataset_emojis", "sad.png"),
    6: os.path.join(cur_path, "Dataset_emojis", "surprised.png"),
}

# FIX 3: Use cv2's built-in haarcascade path
bounding_box = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
show_text = [0]
frame_number = 0


video_path = os.path.join(cur_path, "Dataset_video", "video.mp4")


def show_subject():
    global frame_number, last_frame1

    cap1 = cv2.VideoCapture(video_path)
    if not cap1.isOpened():
        print("Error: Can't open the video file:", video_path)
        return

    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1

    if frame_number >= length:
        print("Video ended.")
        cap1.release()
        return

    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    flag1, frame1 = cap1.read()
    cap1.release()

    if not flag1 or frame1 is None:
        print("Error: Could not read frame.")
        return

    frame1 = cv2.resize(frame1, (600, 500))
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
        )
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(
            frame1, emotion_dict[maxindex],
            (x + 20, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        show_text[0] = maxindex

    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_subject)


def show_avatar():
    emoji_path = emoji_dist[show_text[0]]
    frame2 = cv2.imread(emoji_path)

    if frame2 is not None:
        pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(pic2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain2.configure(image=imgtk2)

    lmain3.configure(
        text=emotion_dict[show_text[0]],
        font=('arial', 45, 'bold')
    )
    lmain2.after(10, show_avatar)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    frame_number = 0

    root = tk.Tk()
    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')

    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    Button(root, text="Quit", fg="red", command=root.destroy,
           font=('arial', 25, 'bold')).pack(side=BOTTOM)

    # FIX 5: Use relative path for external script
    other_code_path = os.path.join(cur_path, "HandGestureRecognize.py")
    os.system(f"python \"{other_code_path}\"")

    # FIX 6: No threads needed — .after() handles scheduling on main loop
    show_subject()
    show_avatar()
    root.mainloop()