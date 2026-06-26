from flask import Flask, render_template, request
import os
import cv2
from utils.preprocess import extract_frames
from utils.detector import analyze_video
from utils.heatmap import generate_heatmap

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HEATMAP_FOLDER = "static"
os.makedirs(HEATMAP_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(path)
            if img is None:
                return "Error loading image"
            frames = [img]
        else:
            frames = extract_frames(path)

        
        result, confidence, fake_scores, real_scores, explanations = analyze_video(frames)

        
        frame_scores = fake_scores + real_scores
        if len(frame_scores) == 0:
            frame_scores = [confidence]

        
        heatmap_path = "static/heatmap.jpg"

        try:
            # Use first frame for heatmap
            generate_heatmap(frames[0])
        except:
            heatmap_path = None

        return render_template(
            'result.html',
            result=result,
            confidence=round(confidence * 100, 2),
            explanations=explanations,
            file_url=path,

            
            fake_count=len(fake_scores),
            real_count=len(real_scores),

            # ✅ Heatmap
            heatmap_path=heatmap_path,

            # ✅ For advanced charts
            frame_scores=frame_scores
        )

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
