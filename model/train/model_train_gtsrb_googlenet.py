import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset_path = "gtsrb"
csv_path = "gtsrb/Train.csv"
annotations = pd.read_csv(csv_path)
# dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# class CustomDataset(Dataset):
#     def __init__(self, csv_path, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_path)
#         self.root_dir = root_dir
#         self.transform = transform
#     def __len__(self):
#         return len(self.annotations)
#     def __getitem__(self, index):
#         img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
#         image = Image.open(img_name)
#         if self.transform:
#             image = self.transform(image)
#         label = self.annotations.iloc[index, 7]
#         return image, label


class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 7])
        image = Image.open(img_name)
        width = self.data_info.iloc[idx, 0]
        height = self.data_info.iloc[idx, 1]
        roi_x1 = self.data_info.iloc[idx, 2]
        roi_y1 = self.data_info.iloc[idx, 3]
        roi_x2 = self.data_info.iloc[idx, 4]
        roi_y2 = self.data_info.iloc[idx, 5]
        class_id = self.data_info.iloc[idx, 6]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'width': width, 'height': height,
                  'roi_x1': roi_x1, 'roi_y1': roi_y1, 'roi_x2': roi_x2, 'roi_y2': roi_y2,
                  'class_id': class_id}

        return sample

class GtsrbCNNModel(nn.Module):
    def __init__(self, num_classes=43):
        super(GtsrbCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


custom_dataset = CustomDataset(root_dir=dataset_path, csv_file=csv_path, transform=transform)

# for i in range(5):  # Print out 5 samples
#     sample = custom_dataset[i]
#     image = sample['image']
#     class_id = sample['class_id']
#
#     plt.figure(figsize=(3, 3))
#     plt.imshow(image.permute(1, 2, 0))  # Permute tensor dimensions for proper display
#     plt.title(f"Class ID: {class_id}")
#     plt.axis('off')
#     plt.show()

train_size = int(0.8 * len(custom_dataset))
print(train_size)
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# model = models.resnet50(pretrained=True)

# model = GtsrbCNNModel()

model = models.googlenet(pretrained=True, progress=False)



num_features = model.fc.in_features
num_classes = 43  # Change this according to your dataset
model.fc = nn.Linear(num_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['image'].to(device), data['class_id'].to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'googlenet_gtsrb.pth')
model = models.googlenet(pretrained=True, progress=False)

num_features = model.fc.in_features
num_classes = 43  # Change this according to your dataset
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load('googlenet_gtsrb.pth'))
model = model.to(device)

correct = 0
total = 0
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data['image'].to(device), data['class_id'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on validation set: %d %%' % (100 * correct / total))










