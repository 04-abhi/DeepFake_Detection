import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from torchvision import transforms, models
from werkzeug.utils import secure_filename

# 🔧 Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🧠 Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model_path = os.path.join(os.getcwd(),"models/resnet_model.pth")
model = ResNetBinaryClassifier().to(device)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded.")
else:
    print("❌ Model not found.")
    exit()

# 🧼 Preprocessing and face detection
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🧠 Actual DeepFake detection logic (corrected)
def check_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60 or w > 300 or h > 300:
                continue

            face = frame[y:y+h, x:x+w]
            face_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                predictions.append(output.item())

    cap.release()

    if predictions:
        avg_score = np.mean(predictions)
        verdict = "Fake" if avg_score < 0.5 else "Real"
        confidence = (1 - avg_score) * 100 if verdict == "Fake" else avg_score * 100
        return f"{verdict} ({confidence:.2f}%)"
    else:
        return "Unknown (0.00%)"

# 🏠 Home route for upload
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = check_deepfake(filepath)
            return redirect(url_for('result', filename=filename, verdict=result))

    return render_template("index.html")

# 📄 Result route
@app.route("/result")
def result():
    filename = request.args.get('filename')
    verdict = request.args.get('verdict')
    return render_template("result.html", filename=filename, verdict=verdict)

# 📁 Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 🚀 Start server
if __name__ == "__main__":
    app.run(debug=True)