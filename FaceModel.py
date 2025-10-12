#imports
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


#making sure the code runs in the correct place
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparamaters
num_classes = 2
num_epochs = 2
batch_size = 10
learning_rate = 0.001


#Importing the Custom data
transform = transforms.Compose([
    transforms.Resize((640,480)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

dir = r"D:\Wiezmann\POC\POC- DATA\FaceModel\Data" 


train_dir = os.path.join(dir, "train")
test_dir   = os.path.join(dir, "test")

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_dataset   = torchvision.datasets.ImageFolder(test_dir,   transform=transform)

train_dataload = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
test_dataload   = torch.utils.data.DataLoader(dataset= test_dataset,   batch_size=batch_size , shuffle=False)


#creating the model
class CNN_classifier(nn.Module):
    def __init__(self,   num_classes):
        super(CNN_classifier, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,64,5, padding= 2, bias = False), nn.BatchNorm2d(64),nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #320 x 240


            nn.Conv2d(64,128,5, padding = 2, bias = False),nn.BatchNorm2d(128), nn.ReLU(inplace=True), #one convelution with padding to maintain shape, followed by an activation function and normalization
            nn.MaxPool2d(2), #160 x 120


            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), #changing the shape of the tensor, into [128,1,1] by avareging the values, then flattening into an 'array' and activating dropout
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), # hidden layer
            nn.Linear(64, num_classes)
        )



    def forward(self, x):
        x = self.step(x)
        return x
    
model = CNN_classifier(num_classes)
model = model.to(device)




#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


#training loop
total_steps =  len(train_dataload)
for epoch in range(num_epochs):
    model.train()
    for i , (images, labels) in enumerate(train_dataload):
        images = images.to(device)
        labels = labels.to(device)


        #foward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backword
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((i+1 ) % 1 == 0):
            print(f'epoch {epoch + 1}, step {i + 1}/{total_steps}, loss = {loss.item():.4f} ')




#evaluting the model
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_dataload:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs , 1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()


acc = 100 * n_correct / n_samples
print (f'accuracy = {acc :.4f}')

#saving the final model
torch.save(model.state_dict(), r"D:\Wiezmann\POC\POC- Models\FaceModel.pth")
