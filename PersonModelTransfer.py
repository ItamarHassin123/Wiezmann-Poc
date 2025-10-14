import torch, torchvision
from PIL import Image
from torchvision import transforms as transforms

#moving the model to the Gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Loading the correct model with pretrained rates and setting it to eval mode
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()


#Detecting whether a person is presant or not
def person_present(img_path,score_thr=0.6):
    with torch.no_grad():
        x = transforms.ToTensor()(Image.open(img_path)).to(device) #opening the image and transforming it into a tensor
        out = model([x])[0] #passing it through the model
        keep = (out['scores'] >= score_thr) & (out['labels'] == 1) #how sure is the model that a person is presant
        return bool(keep.sum().item())


# Example:
print(person_present(r"C:\Users\itama\OneDrive\Desktop\fasten_seat_belt\image_265.jpg"))