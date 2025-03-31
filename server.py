# uvicorn server:app --reload


from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import shutil
import os
import cv2

app = FastAPI()

# YOLO 모델 로드 (메인)
yolo_model_path = "/home/hgyeo/Desktop/1125/runs/segment/train49/weights/best.pt"
yolo_model = YOLO(yolo_model_path)

# Gemma 모델 로드 (서브)
gemma_model_id = "google/gemma-3-4b-it"
gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma_model_id, device_map="auto", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(gemma_model_id, use_fast=True)

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 정적 파일 서빙
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head>
            <title>YOLO + Gemma X-ray Analyzer</title>
        </head>
        <body>
            <h2>이미지 업로드 (YOLO + Gemma 분석)</h2>
            <form action="/analyze" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="분석 시작">
            </form>
        </body>
    </html>
    """

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 이미지 저장
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ YOLO 탐지 수행
    yolo_results = yolo_model.predict(file_location)
    detected_classes = yolo_results[0].names
    boxes = yolo_results[0].boxes

    # YOLO 결과 요약
    yolo_detected = []
    for box in boxes:
        cls = int(box.cls[0])
        name = detected_classes[cls]
        yolo_detected.append(name)

    # 1-2️⃣ YOLO segmentation 이미지 저장
    result_image = yolo_results[0].plot()
    output_image_name = f"{os.path.splitext(file.filename)[0]}_pred.png"
    output_image_path = os.path.join(OUTPUT_DIR, output_image_name)
    cv2.imwrite(output_image_path, result_image)

    # 2️⃣ 항상 Gemma로 보조 설명 추가
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an X-ray security inspection assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": file_location},
                {"type": "text", "text": "If there are any explosive or dangerous items visible in the X-ray image, briefly and clearly describe the name of the item and where it is located. Example:\n- gun: bottom right\n- knife: top left\nIf nothing is detected, simply reply with: 'No threats detected.'"}
            ]
        }

    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(gemma_model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = gemma_model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )
        generation = generation[0][input_len:]

    gemma_description = processor.decode(generation, skip_special_tokens=True)

    return HTMLResponse(content=f"""
        <html>
            <body>
                <h3>YOLO 탐지 결과</h3>
                <p>{', '.join(yolo_detected) if yolo_detected else '탐지된 위협 없음'}</p>
                <h3>Gemma 분석 결과</h3>
                <p>{gemma_description}</p>
                <h3>Segmentation 결과 이미지</h3>
                <img src="/outputs/{output_image_name}" width="600">
                <br><br><a href="/">← 돌아가기</a>
            </body>
        </html>
    """)
