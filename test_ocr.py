import torch
import easyocr
import numpy as np
from PIL import Image

def get_device():
    return torch.cuda.is_available()

def load_model(use_gpu):
    print(f"[OCR] EasyOCR 로드 중 (GPU 사용: {use_gpu})...")
    reader = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
    return reader

def predict(reader, image: Image.Image):
    # EasyOCR은 numpy array(OpenCV format)나 파일 경로, 바이트 배열을 받음
    img_np = np.array(image.convert('RGB'))
    # BGR로 변환하지 않아도 OCR 성능에 큰 지장은 없으나, 보통 RGB/BGR 관계없이 잘 동작함
    
    results = reader.readtext(img_np)
    
    formatted_results = []
    for (bbox, text, prob) in results:
        # bbox의 요소들은 numpy 자료형일 수 있으므로 파이썬 native type으로 변환
        box = [[int(pt[0]), int(pt[1])] for pt in bbox]
        formatted_results.append({
            "text": text,
            "confidence": float(prob),
            "box": box
        })
        
    return {"num_texts": len(formatted_results), "texts": formatted_results}

def main():
    use_gpu = get_device()
    reader = load_model(use_gpu)
    dummy_img = Image.new('RGB', (224, 224), color = 'white')
    print("OCR 테스트 완료. 결과:", predict(reader, dummy_img))

if __name__ == "__main__":
    main()


