import torch
import torch.nn as nn
import os
import numpy as np
import cv2

from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.softmax(x, dim = 1)
        return x


class LanguagePredictor :
    def __init__(self):
        self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        self.model_path = "lpn/lpn_weight.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image):
        """
        이미지 언어 예측
        """
        if isinstance(image, torch.Tensor):
            if image.size == 0 :
                raise ValueError("The provided tensor is empty")
            image = image
        elif isinstance(image, np.ndarray):
            if image.size == 0 :
                raise ValueError("The provided numpy array is empty")
            
            if image.shape[2] != 3 :
                raise ValueError(f"Expected a 3-channel image, but got {image.shape[2]} channels.")

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else :
            image = Image.open(image).convert("RGB")
            
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0)
        else:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        device = next(self.model.parameters()).device
        image = image.to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        index_to_label = {0: "Latin", 1: "Korean", 2: "Japanese", 3: "Chinese"}
        return index_to_label[predicted.item()]
    
    def predict_languages_in_folder(self, folder_path):
        """
        폴더 내 이미지 언어 예측
        """
        image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        predicted_languages = {}
        
        for filename in tqdm(image_files, desc="Predicting languages"):
            image_path = os.path.join(folder_path, filename)
            predicted_language = self.predict(image_path)
            predicted_languages[filename] = predicted_language

        return predicted_languages


if __name__ == "__main__":
    model = LanguagePredictor()
    predicted_languages = model.predict_languages_in_folder("/home/sypark/test/DocLayout-YOLO/outputs/plain_text")
    print(predicted_languages)
