import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
image_model = load_model("hybrid_model.h5")
video_model = load_model("video_detection_model3.h5")

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_frames_from_video(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {i} from {video_path}")
            continue

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    return np.array(frames)

def preprocess_frames(frames):
    frames = frames.astype('float32') / 255.0
    return frames

def predict_image(file_path):
    preprocessed_image = preprocess_image(file_path)
    prediction = image_model.predict(preprocessed_image)
    result = "Deepfake" if prediction < 0.5 else "Real"
    return result

def predict_video(video_path, num_frames=30):
    frames = extract_frames_from_video(video_path, num_frames)
    frames = preprocess_frames(frames)
    frames = frames.reshape((1, num_frames, 224, 224, 3))
    prediction_score = video_model.predict(frames)[0][0]
    threshold = 0.5

    if prediction_score > threshold:
        return "Real"
    else:
        return "Fake"

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/detect", methods=["POST"])
def detect():
    if 'file' not in request.files:
        return render_template("error.html", message="No file uploaded.")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template("error.html", message="No file selected.")
    
    # Determine file type
    if file.filename.endswith(('png', 'jpg', 'jpeg')):
        file_type = 'image'
    elif file.filename.endswith('mp4'):
        file_type = 'video'
    else:
        return render_template("error.html", message="Unsupported file format.")
    
    # Save the uploaded file as 'temp' in the static/temp directory, overwriting existing file
    file_path = os.path.join("static", "temp", "temp" + os.path.splitext(file.filename)[1])
    file.save(file_path)
    
    if file_type == 'image':
        result = predict_image(file_path)
    elif file_type == 'video':
        result = predict_video(file_path)
    
    return render_template("result.html", result=result, file_type=file_type)




if __name__ == "__main__":
    if not os.path.exists(os.path.join("static", "temp")):
        os.makedirs(os.path.join("static", "temp"))
    app.run(debug=True)
