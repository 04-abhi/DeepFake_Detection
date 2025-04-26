import os
import cv2

def extract_frames_from_videos(input_dir, output_dir, every_n_frames=5):
    for label in ['real', 'fake']:
        input_path = os.path.join(input_dir, label)
        output_path = os.path.join(output_dir, label)
        os.makedirs(output_path, exist_ok=True)

        for video_file in os.listdir(input_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            video_path = os.path.join(input_path, video_file)
            cap = cv2.VideoCapture(video_path)
            count = 0
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if count % every_n_frames == 0:
                    frame_filename = os.path.join(output_path, f"{os.path.splitext(video_file)[0]}_frame{frame_id}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_id += 1

                count += 1
            cap.release()

if __name__ == "__main__":
    extract_frames_from_videos('dataset', 'extracted_faces')
