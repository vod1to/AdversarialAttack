import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchattacks
import numpy as np
import cv2
from tqdm import tqdm
import os

class VGGFace(nn.Module):
    """VGG-Face implementation"""
    def __init__(self):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)
    def get_features(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
class VGGAttackFramework:
    def __init__(self, data_dir, model_path, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        # Initialize model
        self.model = VGGFace().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()        
    def prepare_pairs(self):
        pairs = []
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Same person pairs
        for person in classes:
            person_dir = os.path.join(self.data_dir, person)
            images = os.listdir(person_dir)
            if len(images) >= 2:
                img1 = os.path.join(person_dir, images[0])
                img2 = os.path.join(person_dir, images[1])
                pairs.append((img1, img2, 1))
        
        # Different person pairs
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
        return pairs

    def verify_pair(self, img1_path, img2_path, threshold=1.2):
        # Read and preprocess images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        # Convert to torch tensor and normalize
        img1 = torch.Tensor(img1).float().permute(2, 0, 1).view(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).view(1, 3, 224, 224)
        
        # VGG Face mean subtraction
        mean = torch.Tensor(np.array([129.1863/255.0, 104.7624/255.0, 93.5940/255.0])).double().view(1, 3, 1, 1)

        img1 -= mean
        img2 -= mean
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Get features from fc7 layer
        self.model.eval()
        with torch.no_grad():
            # Forward pass until fc7 for both images
            feat1 = self.model.get_features(img1)
            feat2 = self.model.get_features(img2)
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            # Compute L2 distance
            l2_distance = torch.norm(feat1 - feat2, p=2).item()
            return l2_distance < threshold, l2_distance
    def run_evaluation(self):
        results = {}
        # Clean performance
        print("Evaluating clean performance...")
        clean_results = {'true_positive': 0, 'true_negative': 0,
                        'false_positive': 0, 'false_negative': 0}
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            is_match, distance = self.verify_pair(img1_path, img2_path) 
            if label == 1:
                if is_match: clean_results['true_positive'] += 1
                else: clean_results['false_negative'] += 1
            else:
                if is_match: clean_results['false_positive'] += 1
                else: clean_results['true_negative'] += 1
        
        total = sum(clean_results.values())
        results['clean'] = {
            'accuracy': (clean_results['true_positive'] + clean_results['true_negative']) / total,
            'far': clean_results['false_positive'] / (clean_results['false_positive'] + clean_results['true_negative']),
            'frr': clean_results['false_negative'] / (clean_results['false_negative'] + clean_results['true_positive'])
        }
        return results

if __name__ == "__main__":
    framework = VGGAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_path='E:/AdversarialAttack-2/VGGFace/vgg_face_dag.pth'
    )
    results = framework.run_evaluation()
    
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")