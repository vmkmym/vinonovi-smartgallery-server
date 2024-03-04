from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
import os
from googletrans import Translator
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import onnxruntime as rt
import numpy as np


app = Flask(__name__)

sess = rt.InferenceSession("./blip_model.onnx")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
folder_path = 'C:\\Users\\cute7\\OneDrive\\Desktop\\openvino_flask\\bts\\bts'

def filter_images(folder_path, text):
    english_question = text
    
    filtered_image_paths = []
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"폴더에서 {len(image_paths)}개의 이미지를 찾았습니다.")

    for image_path in image_paths:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, english_question, return_tensors="pt")
        output = model.generate(**inputs, max_length=30)
        answer = processor.decode(output[0], skip_special_tokens=True)

        print(f"이미지: {image_path}, 답변: {answer}")

        if answer.lower() == "yes":
            filtered_image_paths.append(image_path)

    print(f"{len(filtered_image_paths)}개의 이미지가 필터링되었습니다.")

    return filtered_image_paths


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/get_image', methods=['POST'])
def get_image():
    print("😊 ", "+"*35, "열렸따 서버!", "+"*35, "😊")
    if request.method == 'POST':
        text = request.form['text']

        file_paths = filter_images(folder_path, text)

        if not file_paths:
            return jsonify({"error": "No images found"}), 404
        
        image_uris = [f"file://{file_path}" for file_path in file_paths]
        print(image_uris)
        return jsonify({"image_uris": image_uris})
    
if __name__ == "__main__":
    app.run(host='192.168.1.108', port=5000, debug=True)
    print("😊 ", "+"*35, "닫혔따 서버!", "+"*35, "😊")