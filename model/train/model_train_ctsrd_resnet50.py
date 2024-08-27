import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import pandas as pd
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset_path = "images"
csv_path = "annotations.csv"
annotations = pd.read_csv(csv_path)
# dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

class CustomDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.annotations.iloc[index, 7]
        return image, label

# Create an instance of your custom dataset
custom_dataset = CustomDataset(csv_path=csv_path, root_dir=dataset_path, transform=transform)

train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

resnet_model = models.resnet50(pretrained=True)

num_classes = 58
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10  # 根据需要调整
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
resnet_model.to(device)


# for epoch in range(num_epochs):
#     resnet_model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = resnet_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#     epoch_loss = running_loss / len(train_dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#
# # 训练结束后保存模型
# torch.save(resnet_model.state_dict(), 'resnet50_ctsrd.pth')

resnet_model.load_state_dict(torch.load('resnet50_ctsrd.pth'))
resnet_model.eval()  # Set the model to evaluation mode
def get_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Assuming you have a validation loader val_loader
val_accuracy = get_accuracy(resnet_model, val_loader)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')


