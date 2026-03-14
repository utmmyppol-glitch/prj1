import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import torchvision.transforms as T
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    print(f"[Detection] Faster R-CNN 로드 중 (장치: {device})...")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model = model.to(device)
    model.eval()
    return model, weights.meta["categories"]

def predict(model, categories, device, image: Image.Image, score_threshold=0.6):
    transform = T.ToTensor()
    img_tensor = transform(image.convert("RGB")).to(device) # Shape: (C, H, W)

    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    boxes = predictions["boxes"].cpu()
    labels = predictions["labels"].cpu()
    scores = predictions["scores"].cpu()

    filtered_indices = scores > score_threshold

    filtered_boxes = boxes[filtered_indices].tolist()
    filtered_labels = labels[filtered_indices].tolist()
    filtered_scores = scores[filtered_indices].tolist()

    results = []
    for box, label_idx, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        results.append({
            "label": categories[label_idx],
            "score": score,
            "box": box  # [xmin, ymin, xmax, ymax]
        })
        
    return {"num_detections": len(results), "detections": results}
def main():
    device = get_device()
    model, categories = load_model(device)
    # 간단한 더미 테스트 (실행 확인용)
    dummy_img = Image.new('RGB', (224, 224), color = 'red')
    print("탐지 테스트 완료. 결과:", predict(model, categories, device, dummy_img))

if __name__ == "__main__":
    main()
