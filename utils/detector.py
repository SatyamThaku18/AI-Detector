import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔥 Load Model
model = load_model("model/deepfake_model.h5", compile=False)
print("✅ Model Input Shape:", model.input_shape)

# 🔥 Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- FACE DETECTION ----------------
def get_faces(frame):
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]

# ---------------- TEXTURE ----------------
def texture_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

# ---------------- EDGE ARTIFACT ----------------
def edge_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean()

# ---------------- EXPLANATION ----------------
def generate_explanation(score, texture, edge, face_detected):
    reasons = []

    if score > 0.6:
        reasons.append("High deepfake probability from AI model")

    if texture < 10:
        reasons.append("Unnatural smooth texture (AI generated patterns)")

    if edge < 20:
        reasons.append("Lack of sharp edges (AI smoothing detected)")

    if not face_detected:
        reasons.append("No human face detected")

    if score < 0.4:
        reasons.append("Natural facial features detected")

    if not reasons:
        reasons.append("Mixed signals detected")

    return list(set(reasons))

# ---------------- PREDICT ----------------
def predict_frame(frame):
    if frame is None:
        return "Suspicious", 0.5, ["Invalid frame"]

    faces = get_faces(frame)
    texture = texture_score(frame)
    edge = edge_score(frame)

    input_shape = model.input_shape[1:3]

    preds = []

    # 🔥 If faces found → use faces
    if faces:
        for face in faces:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, input_shape)

            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0][0]
            preds.append(pred)

    # 🔥 If NO face → use full image
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_shape)

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0][0]
        preds.append(pred)

    # 🔥 Average prediction
    avg_pred = np.mean(preds)

    explanation = generate_explanation(avg_pred, texture, edge, bool(faces))

    # 🔥 FINAL SMART DECISION (IMPORTANT FIX)
    score = avg_pred

    # Combine AI + heuristics
    if texture < 9:
        score += 0.1
    if edge < 25:
        score += 0.1

    score = min(score, 1.0)

    # 🔥 STRONG CLASSIFICATION
    if score > 0.6:
        return "Fake", float(score), explanation

    elif score < 0.4:
        return "Real", float(1 - score), explanation

    else:
        # Reduce "Suspicious" cases
        if texture < 9 or edge < 25:
            return "Fake", float(score), explanation
        else:
            return "Real", float(1 - score), explanation

# ---------------- VIDEO ----------------
def analyze_video(frames):
    fake_scores = []
    real_scores = []
    explanations = []

    for frame in frames:
        label, conf, exp = predict_frame(frame)

        explanations.extend(exp)

        if label == "Fake":
            fake_scores.append(conf)
        else:
            real_scores.append(conf)

    total = len(frames)

    if total == 0:
        return "No Data", 0, [], [], ["No frames detected"]

    fake_ratio = len(fake_scores) / total
    real_ratio = len(real_scores) / total

    # 🔥 FINAL DECISION (FIXED LOGIC)
    if fake_ratio > real_ratio:
        confidence = fake_ratio
        result = "Fake"
    else:
        confidence = real_ratio
        result = "Real"

    return result, confidence, fake_scores, real_scores, explanations
