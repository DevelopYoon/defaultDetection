from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from yolov5.detect import run
from pathlib import Path
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def detector():
    if request.method == 'GET':
        return render_template('detect.html')
    elif request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        labels = 'static/images/labels/'+Path(file.filename).stem+".txt"
        
        if os.path.exists(labels): 
            os.remove(labels)
            print("지워짐")
       
        if file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # first model
            result_image_path = run(weights="yolov5/mode/leak_yolov5x.pt", source=file_path, data="yolov5/data/data.yaml")
            result_image_path = result_image_path.replace("\\", "/")
            result_image_path = "static/images/" + result_image_path

            # second model
            resultImage = run(weights="yolov5/mode/rebar_yolov5x.pt",source=result_image_path, data="yolov5/data/rebar.yaml")
            resultImage = resultImage.replace("\\", "/")
            resultImage = "static/images/" + resultImage

            # third model
            resultImage = run(weights="yolov5/mode/crack_yolov5x.pt",source=result_image_path, data="yolov5/data/coco128.yaml")
            resultImage = resultImage.replace("\\", "/")
            resultImage = "static/images/" + resultImage

            # fourth model
            resultImage = run(weights="yolov5/mode/peel_yolov5x.pt",source=result_image_path, data="yolov5/data/peel.yaml")
            resultImage = resultImage.replace("\\", "/")
            resultImage = "static/images/" + resultImage

            # fifth model
            resultImage = run(weights="yolov5/mode/white_yolov5x.pt",source=result_image_path, data="yolov5/data/white.yaml")
            resultImage = resultImage.replace("\\", "/")
            resultImage = "static/images/" + resultImage

            return jsonify({'image_path': resultImage})

    return jsonify({'error': 'No file uploaded'}), 400


PROCESSED_IMAGE_PATH = 'static/processed_images/'
app.config['UPLOAD_FOLDER'] = PROCESSED_IMAGE_PATH



@app.route('/get_detection_labels/<image_name>', methods=['GET'])
def get_detection_labels(image_name):
    try:
        # 파일 이름에서 확장자 제거
        base_name = os.path.splitext(image_name)[0]
        
        # 실제 라벨 파일 경로 확인 (디버깅용)
        labels_dir = 'static/images/labels'
        label_path = os.path.join(labels_dir, f'{base_name}.txt')
        print(f"Looking for label file at: {label_path}")  # 디버깅용 출력
        
        # 디렉토리 존재 여부 확인
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}")  # 디버깅용 출력
            return jsonify({
                'status': 'error',
                'message': 'Labels directory not found'
            }), 404
        
        # 파일 존재 여부 확인
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")  # 디버깅용 출력
            return jsonify({
                'status': 'error',
                'message': f'라벨 파일을 찾을 수 없습니다: {base_name}.txt'
            }), 404
            
        # 라벨 데이터 읽기
        detections = []
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 6:
                    detection = {
                        'class_id': int(float(values[0])),
                        'bbox': {
                            'x_center': float(values[1]),
                            'y_center': float(values[2]),
                            'width': float(values[3]),
                            'height': float(values[4]),
                            'accuracy': float(values[5])
                        }
                    }
                    detections.append(detection)
                    print(f"Added detection: {detection}")  # 디버깅용 출력
        
        print(f"Total detections found: {len(detections)}")  # 디버깅용 출력
        
        return jsonify({
            'status': 'success',
            'image_name': image_name,
            'total_detections': len(detections),
            'detections': detections
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")  # 디버깅용 출력
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/onboard')
def coex():
    return render_template('onboard.html')

@app.route('/test')
def test():
    return render_template('test.html')

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_IMAGE_PATH):
        os.makedirs(PROCESSED_IMAGE_PATH)
    app.run(debug=True, host='0.0.0.0', port=22426)