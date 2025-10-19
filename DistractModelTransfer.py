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
num_classes   = 10
batch_size    = 32 
learning_rate = 0.0001
num_epochs    = 10 

#Importing the Custom data 
dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(dir, "POC- DATA", "Distraction Model" ,"train")
val_dir = os.path.join(dir,"POC- DATA", "Distraction Model" ,"val")  


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



train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
val_dataset   = torchvision.datasets.ImageFolder(val_dir,   transform=val_tf)

train_dataload = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_dataload   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

#creating the model 
from torchvision.models import resnet18, ResNet18_Weights #the model we'll transfer learn
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #importing the model
feats = model.fc.in_features #number of final features
model.fc = nn.Linear(feats, num_classes) # adding last layer
model = model.to(device)
#loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) #label smoothing improves confidence 
optimizer= torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) #weight decay improves training by making weights smaller generally
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma= 0.1) #learning rate schedualer


#AMP 
scaler = torch.amp.GradScaler(device) #scaler

#evaluate accuracy on val set
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
            n_samples += labels.size(0)
            n_correct += (prediction == labels).sum().item()
    return 100.0 * n_correct / n_samples if n_samples else 0.0

#training loop
def train(model, criterion, optimizer, scheduler, num_epochs):
    total_steps = len(train_dataload)
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataload):
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 50 == 0:
                print(f'epoch {epoch + 1}, step {i + 1}/{total_steps}, loss = {loss.item():.4f}')

        scheduler.step()
        acc = evaluate_accuracy(model)
        print(f'Epoch {epoch + 1}: val accuracy = {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    print(f'Best val accuracy: {best_acc:.2f}%')
    model.load_state_dict(best_state)
    return model

#train
model = train(model, criterion, optimizer, lr_schedular, num_epochs)
#saving the final model
torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "DistractModelTransfer.pth"))



