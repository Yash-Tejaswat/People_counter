from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
import sys
sys.path.append('yolov5')

from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.utils.datasets import letterbox

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        live_feed = 'live_feed' in request.form

        if live_feed:
            video_path = 0  # Camera index for live feed
        else:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)

            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

        result_img_path = run_detection(video_path, live_feed)
        if result_img_path is None:
            return "Error: Unable to process video", 500
        return render_template('index.html', result_img=result_img_path)
    
    return render_template('index.html')

def run_detection(video_path, live_feed=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    out = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(model.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)
        
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = 'static/result.mp4'
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        out.write(frame)
    
    cap.release()
    out.release()
    
    return 'static/result.mp4'

if __name__ == "__main__":
    app.run(debug=True)
