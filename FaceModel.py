#imports
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

#making sure the code runs in the correct place
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparamaters
input_size = 640*480 #photo input size in MNIST data
num_classes = 2
num_epochs = 10
batch_size = 8
learning_rate = 3e-3


#Importing the Custom data
transform = transforms.Compose([
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
    def __init__(self, input_size,  num_classes):
        super(CNN_classifier, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,64,5), #476x636
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #238 x 318

            nn.Conv2d(64,128,5), #234x314
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #117 x 157

            nn.Conv2d(128,256,5), #113 x 153
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #56 x 76

            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),             
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  

        )



    def forward(self, x):
        x = self.step(x)
        return x
    
model = CNN_classifier(input_size, num_classes)
model = model.to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


#training loop
total_steps =  len(train_dataload)
for epoch in range(num_epochs):
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
            print(f'epoch {epoch + 1}, step {i + 1}, loss = {loss.item():.4f} ')




#eval
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

