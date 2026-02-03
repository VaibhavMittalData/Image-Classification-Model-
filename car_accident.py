import torch 
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import numpy as np

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_data = datasets.ImageFolder("pytorch/data/train", transform=transform)
test_data  = datasets.ImageFolder("pytorch/data/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

device ="cuda" if torch.cuda.is_available() else "cpu"
print("using Device: ",device)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,padding=1,stride=1),#128
            nn.ReLU(),
            nn.MaxPool2d(2),#64
            nn.Conv2d(8,16,kernel_size=3,padding=1),#64
            nn.ReLU(),
            nn.MaxPool2d(2),#32
            nn.Conv2d(16,32,kernel_size=3,padding=1,stride=1),#32
            nn.ReLU(),
            nn.MaxPool2d(2),#16
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),#16
            nn.ReLU(),
            nn.MaxPool2d(2),#8
            nn.Conv2d(64,128,kernel_size=3,padding=1,stride=1),#8
            nn.ReLU(),
            nn.MaxPool2d(2)#4

        )

        self.classifier = nn.Sequential(
            nn.Linear(128*4*4,6),
            nn.ReLU(),
            nn.Linear(6,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
model = SimpleCNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(50):
    model.train()
    total_loss =0

    for images,label in train_loader:
        images = images.to(device)
        label  = label.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,label)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f"Epoch {epoch+1}/{50}   Loss: {total_loss:.4f}")

model.eval()

y_true = []
y_pred = []


with torch.no_grad():
    for images,label in test_loader:
        images = images.to(device)
        label = label.to(device)
        
        output = model(images)
        prediction = output.argmax(dim=1)

        
        y_true.extend(label.cpu().numpy())
        y_pred.extend(prediction.cpu().numpy())

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=train_data.classes
))

# model.eval()
# img = cv2.imread(r"E:\ML\pytorch\test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
# img = cv2.resize(img, (128, 128))           

# img = transforms.ToTensor()(img)
# img = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])(img)

# img = img.unsqueeze(0).to(device)   # shape: (1,3,128,128)
# with torch.no_grad():
#     output = model(img)
#     pred = output.argmax(dim=1).item()

# class_names = train_data.classes
# print("Predicted class:", class_names[pred])


