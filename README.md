
# 🔍 X-ray 위협물 탐지 시스템 (YOLO + Gemma3 Multimodal)

> YOLO 모델로 실시간 객체 탐지, 그리고 멀티모달 모델 Gemma3를 서브로 활용해 X-ray 이미지의 위해물품을 보조 분석하는 보안용 웹 애플리케이션입니다.

---

## 📌 프로젝트 목적

- ✅ **YOLOv8-Seg**: X-ray 이미지에서 위험 물체(총, 칼, 가위 등)를 빠르게 탐지하고 segmentation
- ✅ **Gemma3 4B-IT (Google)**: YOLO가 탐지하지 못한 부분을 보완하기 위해 이미지와 텍스트를 함께 이해해 텍스트 설명 생성
- ✅ **FastAPI 서버**: 이미지 업로드 후 탐지 결과 + 분석 요약 텍스트를 웹에서 바로 확인 가능

---

## 🧠 구조 개요

```
사용자 이미지 업로드
        ↓
YOLOv8 → 위험 물체 탐지 + Segmentation 시각화
        ↓
Gemma3 → YOLO와 무관하게 이미지 전체 분석 후 텍스트 보조 설명 생성
        ↓
FastAPI → 결과와 예측 이미지를 클라이언트에 반환
```

---

## ⚙️ 기술 스택

| 기술 / 모델               | 역할 |
|----------------------------|------|
| `YOLOv8-seg`               | X-ray 위협물 탐지 및 시각화 |
| `Gemma3 4B-IT (Google)`    | 텍스트-이미지 멀티모달 보조 설명 |
| `transformers`, `torch`    | 모델 로딩 및 추론 |
| `FastAPI`                  | 웹 서버 및 이미지 처리 |
| `cv2`, `PIL`               | 이미지 입출력 처리 |

---

## 🚀 실행 방법

### 1. YOLO 모델 실행
```python
from ultralytics import YOLO
model = YOLO("runs/segment/train49/weights/best.pt")
results = model.predict(source="uploads/input.png")
```

### 2. FastAPI 서버 실행
```bash
uvicorn server:app --reload
```

- 웹 브라우저에서 `http://127.0.0.1:8000` 접속 후 이미지 업로드

---

## 📂 폴더 구조 예시

```
project/
├── uploads/         ← 업로드된 원본 이미지
├── outputs/         ← YOLO 예측 이미지
├── server.py        ← FastAPI 서버 메인 코드
├── gemma_module.py  ← Gemma3 보조 분석 처리 모듈
```

---

## ✍️ 예시 프롬프트 (Gemma3)

```plaintext
If there are any explosive or dangerous items visible in the X-ray image, 
briefly and clearly describe the name of the item and where it is located. 
Example: 
- gun: bottom right
- knife: top left
If nothing is found, simply reply with: 'No threats detected.'
```

---

## 🖼️ 결과 화면 예시

| YOLO 탐지 결과 | Gemma 보조 설명 |
|----------------|------------------|
| ![yolo](https://velog.velcdn.com/images/hgy9511/post/3c3588ad-e9d7-48cd-90aa-c186b53543f9/image.png) | `Handgun : Center of the image` |

---

## 🚀 모델 응답 속도 차이: 왜 FastAPI 서버 방식이 더 빠를까?

| 항목                         | 단독 실행 스크립트               | FastAPI 서버 방식            |
|------------------------------|-----------------------------------|-------------------------------|
| 모델 로딩 방식                | 실행할 때마다 새로 로드          | 서버 시작 시 1번만 로드       |
| 응답 속도                    | 느림 (최초 실행 시 수십 초~1분) | 빠름 (3~5초 내외)            |
| 메모리 효율                  | 매번 초기화                      | 메모리에 계속 유지            |
| 실시간 서비스 적합성         | ❌ 실험용, 테스트에 적합         | ✅ 실시간 요청에 적합         |

### 💡 왜 차이가 날까?

- Python 스크립트로 실행할 경우, 모델을 **매번 디스크에서 메모리로 로드**하기 때문에 시간이 오래 걸림
- FastAPI나 Ollama처럼 **서버가 모델을 미리 띄워둔 상태**라면,
  → 클라이언트 요청이 들어오면 즉시 응답 가능
- 이는 마치 카페에서 매번 원두를 갈아 커피를 내리는 것 vs 이미 준비된 커피를 서빙하는 차이와 같음 ☕️

---

## ✅ 향후 개선사항

- [ ] 다중 이미지 업로드 및 배치 처리 지원
- [ ] 추론 속도 개선 (GPU 캐싱, 토크나이저 병렬화 등)
- [ ] 탐지 결과를 JSON으로 저장해 로그화
- [ ] 모델 별 confidence score 비교
