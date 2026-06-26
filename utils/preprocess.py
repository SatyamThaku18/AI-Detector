import cv2

def extract_frames(video_path, skip=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames
