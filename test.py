import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms

#creating the model
class CNN_Distract(nn.Module):
    def __init__(self,   num_classes):
        super(CNN_Distract, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,64,5, padding= 2, bias = False), nn.BatchNorm2d(64),nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #320 x 240


            nn.Conv2d(64,128,5, padding = 2, bias = False),nn.BatchNorm2d(128), nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #160 x 120

            nn.Conv2d(128,256,5, padding = 2, bias = False),nn.BatchNorm2d(256), nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #80 x 60

            nn.Conv2d(256,512,5, padding = 2, bias = False),nn.BatchNorm2d(512), nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #80 x 60


            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), #changing the shape of the tensor, into [128,1,1] by avareging the values, then flattening into an 'array' and activating dropout
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), # hidden layer
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), # hidden layer
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), # hidden layer
            nn.Linear(64, num_classes)
        )



    def forward(self, x):
        x = self.step(x)
        return x
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN_Distract(2).to(device)
state = torch.load(r"D:\Wiezmann\POC\POC- Models\FaceModel.pth", map_location=device)
model.load_state_dict(state)
model.eval()



transform = transforms.Compose([
    transforms.Resize((640, 480)),  # (H, W)
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

with torch.no_grad():
    #path = r"C:\Users\itama\OneDrive\Desktop\fasten_seat_belt\image_189.jpg"
    path = r"D:\Wiezmann\POC\POC- DATA\FaceModel\Data\train\Safe Driving\img_4013.jpg"
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)   
    logits = model(x)                            # [1,2]
    pred_idx = logits.argmax(dim=1).item()
    print("prediction:", pred_idx)
