import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm

# ✅ Set paths
model_path = os.path.join(os.getcwd(),"models/resnet_model.pth")
video_dirs = {
    'real': os.path.join(os.getcwd(),"dataset/real"),
    'fake': os.path.join(os.getcwd(),"dataset/fake")
}
result_file_path = os.path.join(os.getcwd(),"testResult.txt")

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define model
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),                # Help prevent overfitting
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = ResNetBinaryClassifier().to(device)

# ✅ Load model
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded.")
else:
    print("❌ Model checkpoint not found!")
    exit()

# ✅ Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Inference for one video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
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
        label = "Fake" if avg_score < 0.5 else "Real"
        return label, avg_score
    else:
        return "Unknown", 0.0

# ✅ Clear result file with UTF-8 encoding
with open(result_file_path, "w", encoding="utf-8") as f:
    f.write("=== DeepFake Detection Test Results ===\n")

# ✅ Run prediction on all videos
for label, folder in video_dirs.items():
    print(f"\n📁 Scanning '{label.upper()}' folder...")

    for video_file in tqdm(os.listdir(folder)):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(folder, video_file)
        prediction, score = predict_video(video_path)

        result = (
            f"🎞️ Video: {video_file}\n"
            f"Actual:   {label.capitalize()}\n"
            f"Predicted:{prediction}\n"
            f"Confidence Score: {score:.4f}\n"
            + "-"*40 + "\n"
        )

        print(result)

        with open(result_file_path, "a", encoding="utf-8") as f:
            f.write(result)

print("\n✅ All predictions done. Results saved to testResult.txt.")
