import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from datetime import datetime

# Paths
model_path = os.path.join(os.getcwd(),"models/resnet_model.pth")
upload_dir = os.path.join(os.getcwd(),"upload")
video_path = os.path.join(os.getcwd(),"input_video.mp4")        #Change this to your video path

# Setup
os.makedirs(upload_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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

model = ResNetBinaryClassifier().to(device)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded.")
else:
    print("❌ Model not found!")
    exit()

# Face transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Predict and save faces
def process_and_save_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    saved_faces = 0
    frame_count = 0

    while cap.isOpened() and saved_faces < 10:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60 or w > 300 or h > 300:
                continue

            face_img = frame[y:y+h, x:x+w]
            input_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_label = "fake" if output.item() > 0.5 else "real"

            # Save face image with label
            filename = f"{pred_label}_{saved_faces+1}.jpg"
            cv2.imwrite(os.path.join(upload_dir, filename), face_img)
            print(f"💾 Saved: {filename} | Score: {output.item():.4f}")

            saved_faces += 1
            if saved_faces >= 10:
                break

    cap.release()
    print(f"\n✅ Saved {saved_faces} face predictions to '{upload_dir}'")

# Run it
if __name__ == "__main__":
    process_and_save_faces(video_path)
