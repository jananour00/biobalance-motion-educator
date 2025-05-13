import os
from flask import Flask, render_template, request, send_from_directory
import cv2
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل نموذج YOLOv8 محليًا
model_path = "yolov8n-pose.pt"  # تأكد من وجود هذا الملف في نفس المجلد أو ضع المسار الكامل

try:
    model = YOLO(model_path)
    print(f"✅ تم تحميل النموذج من: {model_path}")
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    exit()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        if video:
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")

            # حفظ الفيديو
            video.save(input_path)
            print(f"✔️ تم حفظ الفيديو في: {input_path}")

            # تحليل الفيديو
            try:
                detect_video(input_path, output_path)
                print(f"✔️ تم حفظ الفيديو الناتج في: {output_path}")
            except Exception as e:
                print(f"❌ خطأ أثناء تحليل الفيديو: {e}")
                import traceback
                traceback.print_exc()

            return render_template("index.html", video_path="output.mp4")

    return render_template("index.html", video_path=None)

def detect_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()  # رسم النتائج على الإطار

        out.write(annotated_frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
