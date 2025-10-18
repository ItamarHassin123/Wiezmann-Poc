import torch, torchvision
from PIL import Image
from torchvision import transforms as transforms
import os
import torch.nn as nn
import torchvision.transforms as transforms
import copy

#moving the model to the Gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'



#Person Presant model
#Loading the correct model with pretrained rates and setting it to eval mode
person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()


#Distract classification Model
#Custom Model:
class CNN_Distract(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Distract, self).__init__()
        self.step = nn.Sequential(
            #first conv layer
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #112x112


            #Second conv layer
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #56x56

            #Third conv layer
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #28x28


            #Fourth conv layer
            nn.Conv2d(128,256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            
            
            nn.AdaptiveAvgPool2d(1),  #turns the outputted tensor into a 256,1,1
            nn.Flatten(), #turns the tensor into a [256]
            nn.Dropout(0.2), #remove 20 precent of neurons
            nn.Linear(256, num_classes) #final linear layer
        )

    def forward(self, x):
        x = self.step(x)
        return x
    

#Transforms
val_tf = transforms.Compose([
    transforms.Resize(256), #resizes
    transforms.CenterCrop(224), #crops to size
    transforms.ToTensor(),#turns the image into a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),#normalizes the image based on imageNet stats
])










#Transfer model:
from torchvision.models import resnet18, ResNet18_Weights 
Distract_Model_Transfer = resnet18() 
feats = Distract_Model_Transfer.fc.in_features #number of final features
Distract_Model_Transfer.fc = nn.Linear(feats, 10) # adding last layer
Distract_Model_Transfer.load_state_dict(torch.load(r"D:\Wiezmann\POC\POC- Models\DistractModelTransfer.pth"))
Distract_Model_Transfer.eval()


#Custom Model:
Distract_model_custom = CNN_Distract(10)
Distract_model_custom.load_state_dict(torch.load(r"D:\Wiezmann\POC\POC- Models\DistractModel1.0.pth"))
Distract_model_custom.eval()






#Decector functions

#Distraction
def ClassifyDistract(img_path,score_thr=0.6):
    with torch.no_grad():
        img = Image.open(img_path)
        x = val_tf(img).unsqueeze(0).to(device)  #opening the image and transforming it into a tensor
        Distract_model_custom.to(device).eval()
        logits = Distract_model_custom(x)  #the results        
        pred = logits.argmax(dim=1)        #the chosen class   

        return int(pred.item())   
        

print(ClassifyDistract(r"D:\Wiezmann\POC\POC- DATA\Distraction Model\Data\train\Safe Driving\img_4054.jpg"))

#PersonPresant
def person_present(img_path,score_thr=0.6):
    with torch.no_grad():
        x = transforms.ToTensor()(Image.open(img_path)).to(device) #opening the image and transforming it into a tensor
        out = person_model([x])[0] #passing it through the model
        keep = (out['scores'] >= score_thr) & (out['labels'] == 1) #how sure is the model that a person is presant
        return bool(keep.sum().item())

print(person_present(r"D:\Wiezmann\POC\POC- DATA\Distraction Model\Data\train\Safe Driving\img_3939.jpg"))