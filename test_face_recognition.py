import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import scipy.spatial.distance

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_models(device):
    print(f"[Face Recognition] 모델 로드 중 (장치: {device})...")
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

def extract_face_embedding(image: Image.Image, mtcnn, resnet, device):
    try:
        img = image.convert('RGB')
    except Exception as e:
        print(f"이미지 변환 오류: {e}")
        return None

    face, prob = mtcnn(img, return_prob=True)
    
    if face is None:
        return None
    
    face_tensor = face.unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = resnet(face_tensor)
        
    return embedding.cpu().numpy()[0]

def predict(mtcnn, resnet, device, image1: Image.Image, image2: Image.Image, threshold=0.8):
    emb1 = extract_face_embedding(image1, mtcnn, resnet, device)
    emb2 = extract_face_embedding(image2, mtcnn, resnet, device)
    
    if emb1 is None or emb2 is None:
        return {"error": "얼굴을 감지하지 못했습니다."}
    
    cosine_distance = scipy.spatial.distance.cosine(emb1, emb2)
    similarity = 1 - cosine_distance
    is_same = similarity >= threshold
    
    return {
        "similarity": float(similarity),
        "threshold": threshold,
        "is_same_person": bool(is_same)
    }

def main():
    device = get_device()
    mtcnn, resnet = load_models(device)
    dummy_img1 = Image.new('RGB', (224, 224), color = 'red')
    dummy_img2 = Image.new('RGB', (224, 224), color = 'blue')
    print("얼굴 인식 테스트 완료 (얼굴 없음 에러 예상):", predict(mtcnn, resnet, device, dummy_img1, dummy_img2))

if __name__ == "__main__":
    main()


