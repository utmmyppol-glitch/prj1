from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import io
import torch
from transformers import pipeline


# 각 모델 모듈 임포트
import test_model as cls_module
import test_detection as det_module
import test_pose_estimation as pose_module
import test_face_recognition as face_module
import test_ocr as ocr_module

# 전역 모델 저장소 (라이프스팬에서 로드)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 시 모델들을 한 번만 로드하여 메모리에 올립니다.
    """
    print("[서버 시작] 딥러닝 모델들을 메모리에 로드합니다. (시간이 조금 걸릴 수 있습니다.)")
    
    # 공통 장치 설정 (전역)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"메인 추론 장치: {device}")
    
    # 1. Classification (MobileNetV2)
    models['cls_device'] = cls_module.get_device()
    models['cls'] = cls_module.load_model(models['cls_device'])
    
    # 2. Object Detection (Faster R-CNN)
    models['det_device'] = det_module.get_device()
    det_model, det_categories = det_module.load_model(models['det_device'])
    models['det'] = det_model
    models['det_categories'] = det_categories
    
    # 3. Pose Estimation (Keypoint R-CNN)
    models['pose_device'] = pose_module.get_device()
    models['pose'] = pose_module.load_model(models['pose_device'])
    
    # 4. Face Recognition (MTCNN + InceptionResnetV1)
    models['face_device'] = face_module.get_device()
    face_mtcnn, face_resnet = face_module.load_models(models['face_device'])
    models['face_mtcnn'] = face_mtcnn
    models['face_resnet'] = face_resnet
    
    # 5. OCR (EasyOCR)
    models['ocr_gpu'] = ocr_module.get_device()
    models['ocr'] = ocr_module.load_model(models['ocr_gpu'])
    
    # 6. Sentiment Analysis (Hugging Face Transformers)
    print("감정 분석 모델 로드 중...")
    models['sentiment'] = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    print("[서버 시작 완료] 모든 모델이 정상적으로 로드되었습니다.")
    
    yield
    
    # 서버 종료 시 정리 작업
    print("[서버 종료] 메모리를 정리합니다.")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# FastAPI 앱 생성
app = FastAPI(title="Deep Learning Models API", lifespan=lifespan)

# 유틸리티 함수: 업로드된 파일을 PIL 이미지로 변환
async def load_image_from_upload(file: UploadFile) -> Image.Image:
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")


@app.get("/")
async def root():
    return {"message": "딥러닝 API 서버가 정상 동작 중입니다. /docs에서 API 명세를 확인하세요."}


@app.post("/predict/classification")
async def predict_classification(file: UploadFile = File(...)):
    """가벼운 이미지 분류 (MobileNetV2)"""
    image = await load_image_from_upload(file)
    result = cls_module.predict(models['cls'], models['cls_device'], image)
    return JSONResponse(content=result)


@app.post("/predict/detection")
async def predict_detection(file: UploadFile = File(...), score_threshold: float = Form(0.6)):
    """객체 탐지 (Faster R-CNN)"""
    image = await load_image_from_upload(file)
    result = det_module.predict(
        models['det'], 
        models['det_categories'], 
        models['det_device'], 
        image, 
        score_threshold=score_threshold
    )
    return JSONResponse(content=result)


@app.post("/predict/pose-estimation")
async def predict_pose_estimation(file: UploadFile = File(...), score_threshold: float = Form(0.9)):
    """사람 자세 추정 (Keypoint R-CNN)"""
    image = await load_image_from_upload(file)
    result = pose_module.predict(
        models['pose'], 
        models['pose_device'], 
        image, 
        score_threshold=score_threshold
    )
    return JSONResponse(content=result)


@app.post("/predict/face-recognition")
async def predict_face_recognition(
    img1: UploadFile = File(...), 
    img2: UploadFile = File(...), 
    threshold: float = Form(0.8)
):
    """얼굴 인식 (동일인 판별 - MTCNN + InceptionResnetV1)"""
    image1 = await load_image_from_upload(img1)
    image2 = await load_image_from_upload(img2)
    
    result = face_module.predict(
        models['face_mtcnn'], 
        models['face_resnet'], 
        models['face_device'], 
        image1, 
        image2, 
        threshold=threshold
    )
    
    # 에러가 포함된 딕셔너리가 반환되었을 경우
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return JSONResponse(content=result)


@app.post("/predict/ocr")
async def predict_ocr(file: UploadFile = File(...)):
    """광학 문자 인식 (EasyOCR, 한글 지원)"""
    image = await load_image_from_upload(file)
    result = ocr_module.predict(models['ocr'], image)
    return JSONResponse(content=result)
    

@app.post("/predict/sentiment")
async def predict_sentiment(text: str = Form(...)):
    """텍스트 감정 분석 (Hugging Face Transformers)"""
    # pipeline은 리스트 형태의 결과를 반환하므로 첫 번째 항목 추출
    result = models['sentiment'](text)[0]
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
