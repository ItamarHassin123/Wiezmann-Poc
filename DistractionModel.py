#imports
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy

#making sure the code runs in the correct place
device = ('cuda' if torch.cuda.is_available() else 'cpu')


#hyperparamaters
num_classes = 10
batch_size = 25
learning_rate = 0.001
num_epochs = 12


#Importing the Custom data 
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), #randomly crops the photo to size 224x224, as small as 70% of the image (simulates zoom)
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),#randomly changes image charachtaristics like brightness and contrast, simulates real life changes
    transforms.RandomPerspective(distortion_scale=0.25, p=0.2), #applys a random warp to simulate different camera angle
    transforms.ToTensor(), #turns the image into a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), #normalizes the image based on imageNet stats
])
val_tf = transforms.Compose([
    transforms.Resize(256), #resizes
    transforms.CenterCrop(224), #crops to size
    transforms.ToTensor(),#turns the image into a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),#normalizes the image based on imageNet stats
])

dir = r"D:\Wiezmann\POC\POC- DATA\Distraction Model\Data" 


train_dir = os.path.join(dir, "train")
val_dir   = os.path.join(dir, "val")

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
val_dataset   = torchvision.datasets.ImageFolder(val_dir,   transform=val_tf)

train_dataload = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
val_dataload   = torch.utils.data.DataLoader(dataset= val_dataset,   batch_size=batch_size , shuffle=False)


#creating the model
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
    

model = CNN_Distract(num_classes)
model = model.to(device)




#loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) #label smoothing improves confidence 
optimizer= torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) #weight decay improves training by making weights smaller generally
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma= 0.1) #learning rate schedualer



#Evaluater
def evaluate_accuracy(model):
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for images, labels in val_dataload:
            images = images.to(device)
            labels = labels.long().to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()
    return 100.0 * n_correct / n_samples





#training loop 
def train(model, criterion, optimizer, scheduler, num_epochs):
    total_steps = len(train_dataload)
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict()) #saving the best model

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataload):
            images = images.to(device)
            labels = labels.long().to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'epoch {epoch + 1}, step {i + 1}/{total_steps}, loss = {loss.item():.4f}')
       
        scheduler.step()

        acc = evaluate_accuracy(model)
        print(f'Epoch {epoch + 1}: test accuracy = {acc:.6f}%')

        # save best
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model.state_dict())

    print(f'Best test accuracy: {best_acc:.4f}%')
    model.load_state_dict(best_model)
    return model



model = train(model, criterion, optimizer, lr_schedular, num_epochs)



#saving the final model
torch.save(model.state_dict(), r"D:\Wiezmann\POC\POC- Models\DistractModel2.0.pth")
