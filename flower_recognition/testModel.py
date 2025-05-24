# flower_recognition_model.py
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet152

class FlowerRecognitionModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # 创建 ResNet152 模型
        model = resnet152(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 5)  # 输出层调整为5类
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # 设置为评估模式
        return model

    def preprocess_image(self, image_path):
        # 图像预处理（需与训练时一致）
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # 添加 batch 维度

    def predict(self, image_path):
        # 预测流程
        input_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class = torch.argmax(output).item()
        return predicted_class
