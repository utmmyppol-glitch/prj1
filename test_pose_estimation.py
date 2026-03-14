import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    print(f"[Pose Estimation] Keypoint R-CNN 로드 중 (장치: {device})...")
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model = model.to(device)
    model.eval()
    return model

def predict(model, device, image: Image.Image, score_threshold=0.9):
    transform = T.ToTensor()
    img_tensor = transform(image.convert("RGB")).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes = predictions["boxes"].cpu()
    scores = predictions["scores"].cpu()
    keypoints = predictions["keypoints"].cpu()

    filtered_indices = scores > score_threshold

    filtered_boxes = boxes[filtered_indices].tolist()
    filtered_keypoints = keypoints[filtered_indices].tolist()

    persons = []
    for box, kpts in zip(filtered_boxes, filtered_keypoints):
        persons.append({
            "box": box, # [xmin, ymin, xmax, ymax]
            "keypoints": kpts # [[[x, y, visibility], ...]]
        })

    return {"num_persons": len(persons), "persons": persons}

def main():
    device = get_device()
    model = load_model(device)
    dummy_img = Image.new('RGB', (224, 224), color = 'blue')
    print("포즈 검출 테스트 완료. 결과:", predict(model, device, dummy_img))

if __name__ == "__main__":
    main()
