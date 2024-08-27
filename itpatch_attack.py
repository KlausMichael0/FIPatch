import cv2
import sys
import numpy as np
import csv
import torch.nn.functional as F
from skimage.exposure import is_low_contrast
import matplotlib.pyplot as plt
from pyswarm import pso
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from localization import *
from eot import *


# dataset = 'ctsrd' # ctsrd, gtsrb
dataset = 'gtsrb' # ctsrd, gtsrb
model_name = 'cnn' # ctsrd(resnet50, resnet101, vgg13, vgg16), gtsrb(cnn, inceptionv3, mobilenetv2, googlenet)
perturb_radius = 7
circle_number = 1 # 1, 2, 3, 4, 5
n_restarts = 5
test_img_number = 2000
perturb_coordinates = []
location = False
color_test = False
color_used = (255, 0, 0)
save_name = model_name+'_circle_'+str(circle_number)+'_radius_'+str(perturb_radius)
save_folder = 'fail_img/'+save_name        # Save some samples of failed attacks
os.makedirs(save_folder, exist_ok=True)

if model_name == 'inceptionv3':
    input_size = (299, 299)
else:
    input_size = (32, 32)

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

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

class CustomDataset2(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.data_info)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        try:
            image = Image.open(img_name)
            # return image
        except FileNotFoundError:
            print(f"FileNotFoundError: No such file or directory: '{img_name}'")
            return None, None
        width = self.data_info.iloc[idx, 1]
        height = self.data_info.iloc[idx, 2]
        roi_x1 = self.data_info.iloc[idx, 3]
        roi_y1 = self.data_info.iloc[idx, 4]
        roi_x2 = self.data_info.iloc[idx, 5]
        roi_y2 = self.data_info.iloc[idx, 6]
        if self.transform:
            image = self.transform(image)
        label = self.data_info.iloc[idx, 7]
        sample = {'image': image, 'width': width, 'height': height,
                  'roi_x1': roi_x1, 'roi_y1': roi_y1, 'roi_x2': roi_x2, 'roi_y2': roi_y2,
                  'class_id': label}
        return sample

if dataset == 'ctsrd':
    num_classes = 58
    dataset_path = "dataset/CTSRD"
    csv_path = "dataset/ctsrd.csv"
    custom_dataset = CustomDataset2(csv_path=csv_path, root_dir=dataset_path, transform=transform)
else:
    num_classes = 43
    dataset_path = "dataset/GTSRB"
    csv_path = "dataset/gtsrb.csv"
    custom_dataset = CustomDataset(root_dir=dataset_path, csv_file=csv_path, transform=transform)


# custom_dataset = CustomDataset(csv_path=csv_path, root_dir=dataset_path, transform=transform)
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomInceptionV3, self).__init__()
        self.inception = models.inception_v3(pretrained=True, progress=False)
        self.inception.Conv2d_1a_3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Modify the input layer
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)

    def forward(self, x):
        return self.inception(x)

if model_name == 'resnet101':
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('model/resnet101_ctsrd.pth'))
elif model_name == 'resnet50':
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('model/resnet50_ctsrd.pth'))
elif model_name == 'vgg13':
    model = models.vgg13(pretrained=True, progress=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('model/vgg13_ctsrd.pth'))
elif model_name == 'vgg16':
    model = models.vgg16(pretrained=True, progress=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('model/vgg16_ctsrd.pth'))
elif model_name == 'cnn':
    model = GtsrbCNNModel()
    model.load_state_dict(torch.load('model/cnn_gtsrb.pth'))
elif model_name == 'inceptionv3':
    model = CustomInceptionV3(num_classes=43)
    model.load_state_dict(torch.load('model/inception_v3_gtsrb.pth'))
elif model_name == 'mobilenetv2':
    model = models.mobilenet_v2(pretrained=False)
    num_features = model.classifier[1].in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('model/mobilenet_v2_gtsrb.pth'))
elif model_name == 'googlenet':
    model = models.googlenet(pretrained=True, progress=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('model/googlenet_gtsrb.pth'))

model = model.to(device)

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data['image'].to(device), data['class_id'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test set: %d %%' % (100 * correct / total))
print(len(custom_dataset))


hough_circle_parameters = {
    "dp": 1,
    "minDist": 150,
    "param1": 200,
    "param2": 15,
    "minRadius": 10,
    "maxRadius": 100
}

predicted_original_class = 0

def objective_function_color(params):
    x1, y1, r1, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r1), color_used, thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value

def objective_function_color2(params):
    x1, y1, r1, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r1), color_used, thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function(params):
    x1, y1, r1, r_prime, g_prime, b_prime, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r1), (b_prime, g_prime, r_prime), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function2(params):
    x1, y1, r1, r_prime, g_prime, b_prime, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r1), (b_prime, g_prime, r_prime), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_two_circle(params):
    x1, y1, x2, y2, r, r1, g1, b1, r2, g2, b2, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_two_circle2(params):
    x1, y1, x2, y2, r, r1, g1, b1, r2, g2, b2, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_three_circle(params):
    x1, y1, x2, y2, x3, y3, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_three_circle2(params):
    x1, y1, x2, y2, x3, y3, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_four_circle(params):
    x1, y1, x2, y2, x3, y3, x4, y4, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    cv2.circle(circle, (int(x4), int(y4)), int(r), (b4, g4, r4), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_four_circle2(params):
    x1, y1, x2, y2, x3, y3, x4, y4, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    cv2.circle(circle, (int(x4), int(y4)), int(r), (b4, g4, r4), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_five_circle(params):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    cv2.circle(circle, (int(x4), int(y4)), int(r), (b4, g4, r4), thickness=-1)
    cv2.circle(circle, (int(x5), int(y5)), int(r), (b5, g5, r5), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value


def objective_function_five_circle2(params):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, alpha = params
    circle = np.zeros_like(img_resized)
    cv2.circle(circle, (int(x1), int(y1)), int(r), (b1, g1, r1), thickness=-1)
    cv2.circle(circle, (int(x2), int(y2)), int(r), (b2, g2, r2), thickness=-1)
    cv2.circle(circle, (int(x3), int(y3)), int(r), (b3, g3, r3), thickness=-1)
    cv2.circle(circle, (int(x4), int(y4)), int(r), (b4, g4, r4), thickness=-1)
    cv2.circle(circle, (int(x5), int(y5)), int(r), (b5, g5, r5), thickness=-1)
    result = cv2.bitwise_and(circle, circle, mask=mask_back)
    result2 = cv2.addWeighted(img_resized_copy2, 1 - random_alpha, result, random_alpha, 0)
    result_combined = np.where(result != 0, result2, img_resized_copy2)
    result_pil = Image.fromarray(result_combined)
    input_tensor = transform(result_pil).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    target = False
    if target:
        target_class = 54
        objective_value = -output[0][target_class].item()
        constraint_violation = int(torch.argmax(output[0]).item() != target_class)
        penalty = 1000
        objective_value = objective_value + penalty * constraint_violation
    else:
        original_class = predicted_original_class
        objective_value = output[0][original_class].item() + 0.0001 * np.count_nonzero(result)
    return objective_value

def n_random_restart_pso(objective_function, lb, ub, swarmsize=10, maxiter=500, n_restarts=5):
    best_result = None
    best_value = np.inf
    for _ in range(n_restarts):
        result, value = pso(objective_function, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
        if value < best_value:
            best_result = result
            best_value = value
    return best_result, best_value


alpha = 1.5
beta = 30

total_correct = 0
attack_success = 0
total_samples = 0
test_images = 0
save_images = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data['image'].to(device), data['class_id'].to(device)
        widths, heights = data['width'].to(device), data['height'].to(device)
        roi_x1s, roi_y1s = data['roi_x1'].to(device), data['roi_y1'].to(device)
        roi_x2s, roi_y2s = data['roi_x2'].to(device), data['roi_y2'].to(device)
        if images is None:
            continue
        images_np = images.cpu().numpy()
        images_np = np.transpose(images_np, (0, 2, 3, 1))
        images_np = (images_np * 255).astype(np.uint8)  # [Batch, R, G, B]
        for i in tqdm(range(len(images)), desc='Processing each image', leave=False):
            height_csv = heights[i].item()
            width_csv = widths[i].item()
            roi_x1 = roi_x1s[i].item()
            roi_x2 = roi_x2s[i].item()
            roi_y1 = roi_y1s[i].item()
            roi_y2 = roi_y2s[i].item()
            predicted_original_class = labels[i].item()
            image_original = images_np[i]
            images_cv2 = (image_original).astype(np.uint8)
            images_cv2 = cv2.cvtColor(images_cv2, cv2.COLOR_BGR2RGB)
            # Eot
            images_eot = image_transformation(images_cv2, 10)
            for image in images_eot:
                img_copy = image.copy()
                img_copy2 = image.copy()
                img_copy2 = cv2.cvtColor(img_copy2, cv2.COLOR_BGR2RGB)
                img_denoised = cv2.medianBlur(img_copy, 3)
                if is_low_contrast(img_denoised):
                    img_denoised = contrast_enhance(img_denoised)
                img_resized = img_denoised
                img_resized_copy = img_resized.copy()
                img_resized_copy2 = img_resized.copy()
                img_resized_copy3 = img_resized.copy()
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                edge, canny_th2 = auto_canny(gray, "otsu")
                cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = cnts[0]
                rect = cnt_rect(cnt)
                hough_circle_parameters["param1"] = canny_th2
                circle = cnt_circle(gray, hough_circle_parameters)
                contours = [rect, circle]
                height, width, _ = img_copy.shape
                black_background = np.zeros((height, width, 3), dtype=np.uint8)
                output1 = integrate_circle_rect(rect, circle, cnt)
                color_segmented = color_seg(img_resized)
                cnts_color = cv2.findContours(color_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt_color = cnts_color[0]
                rect_color = cnt_rect(cnt_color)
                hough_circle_parameters["param1"] = 200
                circle_color = cnt_circle(color_segmented, hough_circle_parameters)
                circle_contour = output1
                if len(circle_contour) == 0:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(circle_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(img_resized, center, radius, (0, 255, 0), 2)
                mask = np.zeros_like(img_resized)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                num_white_pixels = np.count_nonzero(mask == 1)
                random_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                random_radius = np.random.randint(0, 50)
                random_alpha = 0.9
                original_size_img = cv2.resize(img_resized, (height_csv, width_csv), interpolation=cv2.INTER_LINEAR)
                mask_as_csv = np.zeros_like(original_size_img)
                cv2.rectangle(mask_as_csv, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), thickness=-1)
                mask_as_csv_resized = cv2.resize(mask_as_csv, input_size, interpolation=cv2.INTER_LINEAR)
                mask_backups = cv2.cvtColor(mask_as_csv_resized, cv2.COLOR_BGR2GRAY)
                ret2, mask_back = cv2.threshold(mask_backups, 0.5, 1, cv2.THRESH_BINARY)
                mask_back = mask_back.astype(np.uint8)
                if circle_number == 1:
                    if color_test:
                        lb = [0, 0, perturb_radius - 0.1, 0.7]
                        ub = [img_resized.shape[1], img_resized.shape[0], perturb_radius + 0.1, 1]
                        best_params, best_value = n_random_restart_pso(objective_function_color, lb, ub, swarmsize=10, maxiter=30,
                                                                       n_restarts=n_restarts)
                        x1, y1, r1, alpha = best_params

                        best_color = color_used
                        best_radius = int(r1)
                        best_alpha = alpha
                        best_x = int(x1)
                        best_y = int(y1)
                        center = (best_x, best_y)
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center, best_radius, best_color, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                            if x1 > 31:
                                x1 = 31
                            if y1 > 31:
                                y1 = 31
                            perturb_coordinates.append([x1, y1])

                        else:
                            best_params, best_value = n_random_restart_pso(objective_function_color2, lb, ub, swarmsize=10,
                                                                           maxiter=30,
                                                                           n_restarts=n_restarts)
                            x1, y1, r, alpha = best_params
                            best_color1 = color_used
                            best_radius = int(r)
                            best_alpha = alpha
                            center1 = (int(x1), int(y1))
                            circle = np.zeros_like(img_resized)
                            cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                            result = cv2.bitwise_and(circle, circle, mask=mask_back)
                            result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                            result_combined = np.where(result != 0, result2, img_resized_copy3)
                            result_combined_pil = Image.fromarray(result_combined)
                            input_tensor = transform(result_combined_pil).unsqueeze(0)
                            input_tensor = input_tensor.to(device)
                            output = model(input_tensor)
                            _, predicted_class = torch.max(output, 1)
                            if predicted_class.item() != predicted_original_class:
                                attack_success += 1
                                if x1 > 31:
                                    x1 = 31
                                if y1 > 31:
                                    y1 = 31
                                perturb_coordinates.append([x1, y1])
                        test_images += 1
                        total_samples += 1
                        print(f'Attack success: {attack_success}')
                        print(f'Now Sample number: {total_samples}')
                        if location:
                            if attack_success == 100:
                                block_counts, block_ratios = count_blocks(perturb_coordinates)
                                plt.imshow(block_counts, cmap='viridis', interpolation='nearest')
                                plt.colorbar(label='Counts')
                                plt.title('Count of Coordinates in Each Block')
                                plt.xlabel('Block X')
                                plt.ylabel('Block Y')
                                for i in range(8):
                                    plt.axhline(i - 0.5, color='black', linewidth=1)
                                    plt.axvline(i - 0.5, color='black', linewidth=1)
                                for i in range(8):
                                    for j in range(8):
                                        plt.text(j, i, f'{block_ratios[i, j]:.2f}', ha='center', va='center', color='white')
                                plt.savefig(save_name + '_heatmap.png')
                                sys.exit()
                        if test_images > (test_img_number - 1):
                            print(f'Attack success: {attack_success}')
                            print(f'Total samples: {total_samples}')
                            asr = attack_success / total_samples
                            print(f'ASR on validation set: {asr * 100:.2f}%')
                            sys.exit()
                    else:
                        lb = [0, 0, perturb_radius-0.1, 0, 0, 0, 0.7]
                        ub = [img_resized.shape[1], img_resized.shape[0], perturb_radius+0.1, 255, 255, 255, 1]  # Upper bounds
                        best_params, best_value = n_random_restart_pso(objective_function, lb, ub, swarmsize=10, maxiter=30,
                                                                   n_restarts=n_restarts)
                        x1, y1, r1, r_prime, g_prime, b_prime, alpha = best_params
                        best_color = (b_prime, g_prime, r_prime)
                        best_radius = int(r1)
                        best_alpha = alpha
                        best_x = int(x1)
                        best_y = int(y1)
                        center = (best_x, best_y)
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center, best_radius, best_color, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                            if x1 > 31:
                                x1=31
                            if y1 >31:
                                y1=31
                            perturb_coordinates.append([x1, y1])
                        else:
                            best_params, best_value = n_random_restart_pso(objective_function2, lb, ub, swarmsize=10, maxiter=30,
                                                                           n_restarts=n_restarts)
                            x1, y1, r, r1, g1, b1, alpha = best_params
                            best_color1 = (b1, g1, r1)
                            best_radius = int(r)
                            best_alpha = alpha
                            center1 = (int(x1), int(y1))
                            circle = np.zeros_like(img_resized)
                            cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                            result = cv2.bitwise_and(circle, circle, mask=mask_back)
                            result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                            result_combined = np.where(result != 0, result2, img_resized_copy3)
                            result_combined_pil = Image.fromarray(result_combined)
                            input_tensor = transform(result_combined_pil).unsqueeze(0)
                            input_tensor = input_tensor.to(device)
                            output = model(input_tensor)
                            _, predicted_class = torch.max(output, 1)
                            if predicted_class.item() != predicted_original_class:
                                attack_success += 1
                                if x1 > 31:
                                    x1 = 31
                                if y1 > 31:
                                    y1 = 31
                                perturb_coordinates.append([x1, y1])
                            else:
                                save_images += 1
                                filename = f"image_{save_images}.png"
                                filepath = os.path.join(save_folder, filename)
                                cv2.imwrite(filepath, result_combined)
                                with open(save_name+'.csv', mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([filename, predicted_original_class])
                        test_images += 1
                        total_samples += 1
                        print(f'Attack success: {attack_success}')
                        print(f'Now Sample number: {total_samples}')
                        if location:
                            if attack_success == 100:
                                block_counts, block_ratios = count_blocks(perturb_coordinates)
                                plt.imshow(block_counts, cmap='viridis', interpolation='nearest')
                                plt.colorbar(label='Counts')
                                plt.title('Count of Coordinates in Each Block')
                                plt.xlabel('Block X')
                                plt.ylabel('Block Y')
                                for i in range(8):
                                    plt.axhline(i - 0.5, color='black', linewidth=1)
                                    plt.axvline(i - 0.5, color='black', linewidth=1)
                                for i in range(8):
                                    for j in range(8):
                                        plt.text(j, i, f'{block_ratios[i, j]:.2f}', ha='center', va='center', color='white')
                                plt.savefig(save_name+'_heatmap.png')
                                sys.exit()
                        if test_images > (test_img_number-1):
                            print(f'Attack success: {attack_success}')
                            print(f'Total samples: {total_samples}')
                            asr = attack_success / total_samples
                            print(f'ASR on validation set: {asr * 100:.2f}%')
                            sys.exit()
                elif circle_number == 2:
                    lb = [0, 0, 0, 0, perturb_radius - 0.1, 0, 0, 0, 0, 0, 0, 0.7]
                    ub = [img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0], perturb_radius + 0.1, 255, 255, 255,
                          255, 255, 255, 1]
                    best_params, best_value = n_random_restart_pso(objective_function_two_circle, lb, ub, swarmsize=10, maxiter=30,
                                                                   n_restarts=n_restarts)
                    x1, y1, x2, y2, r, r1, g1, b1, r2, g2, b2, alpha = best_params
                    best_color1 = (b1, g1, r1)
                    best_color2 = (b2, g2, r2)
                    best_radius = int(r)
                    best_alpha = alpha
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    circle = np.zeros_like(img_resized)
                    cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                    cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                    result = cv2.bitwise_and(circle, circle, mask=mask)
                    result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                    result_combined = np.where(result != 0, result2, img_resized_copy3)
                    result_combined_pil = Image.fromarray(result_combined)
                    input_tensor = transform(result_combined_pil).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() != predicted_original_class:
                        attack_success += 1
                    else:
                        best_params, best_value = n_random_restart_pso(objective_function_two_circle2, lb, ub, swarmsize=10,
                                                                       maxiter=30,
                                                                       n_restarts=n_restarts)
                        x1, y1, x2, y2, r, r1, g1, b1, r2, g2, b2, alpha = best_params
                        best_color1 = (b1, g1, r1)
                        best_color2 = (b2, g2, r2)
                        best_radius = int(r)
                        best_alpha = alpha
                        center1 = (int(x1), int(y1))
                        center2 = (int(x2), int(y2))
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                        cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask_back)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                        else:
                            save_images += 1
                            filename = f"image_{save_images}.png"
                            filepath = os.path.join(save_folder, filename)
                            cv2.imwrite(filepath, result_combined)
                            with open(save_name+'.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([filename, predicted_original_class])
                    test_images += 1
                    total_samples += 1
                    print(f'Attack success: {attack_success}')
                    print(f'Now Sample number: {total_samples}')
                    if test_images > (test_img_number-1):
                        print(f'Attack success: {attack_success}')
                        print(f'Total samples: {total_samples}')
                        asr = attack_success / total_samples
                        print(f'ASR on validation set: {asr * 100:.2f}%')
                        sys.exit()
                elif circle_number == 3:
                    lb = [0, 0, 0, 0, 0, 0,  perturb_radius - 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0.7]
                    ub = [img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0],
                          perturb_radius + 0.1, 255, 255, 255, 255, 255, 255,
                          255, 255, 255, 1]
                    best_params, best_value = n_random_restart_pso(objective_function_three_circle, lb, ub, swarmsize=10,
                                                                   maxiter=30,
                                                                   n_restarts=n_restarts)
                    x1, y1, x2, y2, x3, y3, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, alpha = best_params
                    best_color1 = (b1, g1, r1)
                    best_color2 = (b2, g2, r2)
                    best_color3 = (b3, g3, r3)
                    best_radius = int(r)
                    best_alpha = alpha
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    center3 = (int(x3), int(y3))
                    circle = np.zeros_like(img_resized)
                    cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                    cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                    cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                    result = cv2.bitwise_and(circle, circle, mask=mask)
                    result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                    result_combined = np.where(result != 0, result2, img_resized_copy3)
                    result_combined_pil = Image.fromarray(result_combined)
                    input_tensor = transform(result_combined_pil).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() != predicted_original_class:
                        attack_success += 1
                    else:
                        best_params, best_value = n_random_restart_pso(objective_function_three_circle2, lb, ub, swarmsize=10,
                                                                       maxiter=30,
                                                                       n_restarts=n_restarts)
                        x1, y1, x2, y2, x3, y3, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, alpha = best_params

                        best_color1 = (b1, g1, r1)
                        best_color2 = (b2, g2, r2)
                        best_color3 = (b3, g3, r3)
                        best_radius = int(r)
                        best_alpha = alpha
                        center1 = (int(x1), int(y1))
                        center2 = (int(x2), int(y2))
                        center3 = (int(x3), int(y3))
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                        cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                        cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask_back)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                        else:
                            save_images += 1
                            filename = f"image_{save_images}.png"
                            filepath = os.path.join(save_folder, filename)
                            cv2.imwrite(filepath, result_combined)
                            with open(save_name + '.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([filename, predicted_original_class])
                    test_images += 1
                    total_samples += 1
                    print(f'Attack success: {attack_success}')
                    print(f'Now Sample number: {total_samples}')
                    if test_images > (test_img_number-1):
                        print(f'Attack success: {attack_success}')
                        print(f'Total samples: {total_samples}')
                        asr = attack_success / total_samples
                        print(f'ASR on validation set: {asr * 100:.2f}%')
                        sys.exit()
                elif circle_number == 4:
                    lb = [0, 0, 0, 0, 0, 0, 0, 0, perturb_radius - 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0.7]  # Lower bounds for x1, y1, r1, r', g', b', alpha
                    ub = [img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0],
                          img_resized.shape[1], img_resized.shape[0],
                          perturb_radius + 0.1, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                          255, 255, 255, 1]  # Upper bounds
                    best_params, best_value = n_random_restart_pso(objective_function_four_circle, lb, ub, swarmsize=10,
                                                                   maxiter=30,
                                                                   n_restarts=n_restarts)
                    x1, y1, x2, y2, x3, y3, x4, y4, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, alpha = best_params
                    best_color1 = (b1, g1, r1)
                    best_color2 = (b2, g2, r2)
                    best_color3 = (b3, g3, r3)
                    best_color4 = (b4, g4, r4)
                    best_radius = int(r)
                    best_alpha = alpha
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    center3 = (int(x3), int(y3))
                    center4 = (int(x4), int(y4))
                    circle = np.zeros_like(img_resized)
                    cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                    cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                    cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                    cv2.circle(circle, center4, best_radius, best_color4, thickness=-1)
                    result = cv2.bitwise_and(circle, circle, mask=mask)
                    result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                    result_combined = np.where(result != 0, result2, img_resized_copy3)
                    result_combined_pil = Image.fromarray(result_combined)
                    input_tensor = transform(result_combined_pil).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() != predicted_original_class:
                        attack_success += 1
                    else:
                        best_params, best_value = n_random_restart_pso(objective_function_four_circle2, lb, ub,
                                                                       swarmsize=10,
                                                                       maxiter=30,
                                                                       n_restarts=n_restarts)
                        x1, y1, x2, y2, x3, y3, x4, y4, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, alpha = best_params
                        best_color1 = (b1, g1, r1)
                        best_color2 = (b2, g2, r2)
                        best_color3 = (b3, g3, r3)
                        best_color4 = (b4, g4, r4)
                        best_radius = int(r)
                        best_alpha = alpha
                        center1 = (int(x1), int(y1))
                        center2 = (int(x2), int(y2))
                        center3 = (int(x3), int(y3))
                        center4 = (int(x4), int(y4))
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                        cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                        cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                        cv2.circle(circle, center4, best_radius, best_color4, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask_back)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                        else:
                            save_images += 1
                            filename = f"image_{save_images}.png"
                            filepath = os.path.join(save_folder, filename)
                            cv2.imwrite(filepath, result_combined)
                            with open(save_name + '.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([filename, predicted_original_class])
                    test_images += 1
                    total_samples += 1
                    print(f'Attack success: {attack_success}')
                    print(f'Now Sample number: {total_samples}')
                    if test_images > (test_img_number-1):
                        print(f'Attack success: {attack_success}')
                        print(f'Total samples: {total_samples}')
                        asr = attack_success / total_samples
                        print(f'ASR on validation set: {asr * 100:.2f}%')
                        sys.exit()
                elif circle_number == 5:
                    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, perturb_radius - 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0.7]  # Lower bounds for x1, y1, r1, r', g', b', alpha
                    ub = [img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0],
                          img_resized.shape[1], img_resized.shape[0],
                          img_resized.shape[1], img_resized.shape[0],
                          perturb_radius + 0.1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                          255, 255, 255, 1]  # Upper bounds
                    best_params, best_value = n_random_restart_pso(objective_function_five_circle, lb, ub, swarmsize=10,
                                                                   maxiter=30,
                                                                   n_restarts=n_restarts)
                    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, alpha = best_params
                    best_color1 = (b1, g1, r1)
                    best_color2 = (b2, g2, r2)
                    best_color3 = (b3, g3, r3)
                    best_color4 = (b4, g4, r4)
                    best_color5 = (b5, g5, r5)
                    best_radius = int(r)
                    best_alpha = alpha
                    center1 = (int(x1), int(y1))
                    center2 = (int(x2), int(y2))
                    center3 = (int(x3), int(y3))
                    center4 = (int(x4), int(y4))
                    center5 = (int(x5), int(y5))
                    circle = np.zeros_like(img_resized)
                    cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                    cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                    cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                    cv2.circle(circle, center4, best_radius, best_color4, thickness=-1)
                    cv2.circle(circle, center5, best_radius, best_color5, thickness=-1)
                    result = cv2.bitwise_and(circle, circle, mask=mask)
                    result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                    result_combined = np.where(result != 0, result2, img_resized_copy3)
                    result_combined_pil = Image.fromarray(result_combined)
                    input_tensor = transform(result_combined_pil).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() != predicted_original_class:
                        attack_success += 1
                    else:
                        best_params, best_value = n_random_restart_pso(objective_function_five_circle2, lb, ub,
                                                                       swarmsize=10,
                                                                       maxiter=30,
                                                                       n_restarts=n_restarts)
                        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, r, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, alpha = best_params
                        best_color1 = (b1, g1, r1)
                        best_color2 = (b2, g2, r2)
                        best_color3 = (b3, g3, r3)
                        best_color4 = (b4, g4, r4)
                        best_color5 = (b5, g5, r5)
                        best_radius = int(r)
                        best_alpha = alpha
                        center1 = (int(x1), int(y1))
                        center2 = (int(x2), int(y2))
                        center3 = (int(x3), int(y3))
                        center4 = (int(x4), int(y4))
                        center5 = (int(x5), int(y5))
                        circle = np.zeros_like(img_resized)
                        cv2.circle(circle, center1, best_radius, best_color1, thickness=-1)
                        cv2.circle(circle, center2, best_radius, best_color2, thickness=-1)
                        cv2.circle(circle, center3, best_radius, best_color3, thickness=-1)
                        cv2.circle(circle, center4, best_radius, best_color4, thickness=-1)
                        cv2.circle(circle, center5, best_radius, best_color5, thickness=-1)
                        result = cv2.bitwise_and(circle, circle, mask=mask_back)
                        result2 = cv2.addWeighted(img_resized_copy3, 1 - best_alpha, result, best_alpha, 0)
                        result_combined = np.where(result != 0, result2, img_resized_copy3)
                        result_combined_pil = Image.fromarray(result_combined)
                        input_tensor = transform(result_combined_pil).unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)
                        if predicted_class.item() != predicted_original_class:
                            attack_success += 1
                        else:
                            save_images += 1
                            filename = f"image_{save_images}.png"
                            filepath = os.path.join(save_folder, filename)
                            cv2.imwrite(filepath, result_combined)
                            with open(save_name + '.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([filename, predicted_original_class])
                    test_images += 1
                    total_samples += 1
                    print(f'Attack success: {attack_success}')
                    print(f'Now Sample number: {total_samples}')
                    if test_images > (test_img_number-1):
                        print(f'Attack success: {attack_success}')
                        print(f'Total samples: {total_samples}')
                        asr = attack_success / total_samples
                        print(f'ASR on validation set: {asr * 100:.2f}%')
                        sys.exit()
