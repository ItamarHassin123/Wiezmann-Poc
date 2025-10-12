import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms

class CNN_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,64,5, padding=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,5, padding=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.step(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN_classifier(2).to(device)
state = torch.load(r"D:\Wiezmann\POC\POC- Models\FaceModel.pth", map_location=device)
model.load_state_dict(state)
model.eval()



transform = transforms.Compose([
    transforms.Resize((640, 480)),  # (H, W)
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

with torch.no_grad():
    path = r"C:\Users\itama\OneDrive\Desktop\fasten_seat_belt\image_472.jpg"
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)   
    logits = model(x)                            # [1,2]
    pred_idx = logits.argmax(dim=1).item()
    print("prediction:", logits)
