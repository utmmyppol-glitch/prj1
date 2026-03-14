import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    print(f"[Classification] MobileNetV2 모델 로드 중 (장치: {device})...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    return model

def predict(model, device, image: Image.Image):
    # ImageNet 표준 전처리 적용 (실제 이미지 사용 시 필수)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        
    predicted_class = torch.argmax(output, dim=1).item()
    
    # 텐서플로우/파이토치에서 기본 제공되는 ImageNet 클래스 매핑(단순화를 위해 여기선 인덱스와 점수만 반환)
    # 필요시 추가 가능
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    score = probabilities[predicted_class].item()
    
    return {
        "class_index": predicted_class,
        "confidence": score
    }

def main():
    device = get_device()
    model = load_model(device)
    # 더미 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("분류 테스트 완료. 예측 인덱스:", torch.argmax(output, dim=1).item())

if __name__ == "__main__":
    main()
