import os
import cv2
from tqdm import tqdm

# Paths
input_dirs = {
    'real': os.path.join(os.getcwd(),"extracted_frames/real"),
    'fake': os.path.join(os.getcwd(),"extracted_frames/fake")
}
output_dirs = {
    'real': os.path.join(os.getcwd(),"extracted_faces/real"),
    'fake': os.path.join(os.getcwd(),"extracted_faces/fake")
}

# Create output directories if they don't exist
for key in output_dirs:
    os.makedirs(output_dirs[key], exist_ok=True)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set minimum and maximum face size (in pixels)
MIN_FACE_SIZE = 60     # Minimum width/height
MAX_FACE_SIZE = 300    # Maximum width/height

def preprocess_faces(input_dir, output_dir):
    for frame_file in tqdm(os.listdir(input_dir)):
        frame_path = os.path.join(input_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            # Filter out faces that are too small or too large
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE or w > MAX_FACE_SIZE or h > MAX_FACE_SIZE:
                continue

            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            out_filename = f"{frame_file.split('.')[0]}_face{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_filename), face)

# Run face preprocessing for both real and fake
for label in ['real', 'fake']:
    preprocess_faces(input_dirs[label], output_dirs[label])
