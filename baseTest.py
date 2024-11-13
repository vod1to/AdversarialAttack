import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image

def load_model(model_path, num_classes):
    """Load the trained model"""
    model = models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def get_random_normal_images(dataset_path, num_images):
    all_images = []
    for root, dirs,  files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.basename(root)
                image_path = os.path.join(root, file)
                all_images.append((image_path, class_name))
    
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    return selected_images

def predict_image(model, image_path, transform, device):
    """Make prediction for a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model_path = "outputs/ResNet18Trained.pth" 
    dataset_path = "E:/lfw/lfw-py/lfw_funneled"  
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    num_classes = len(dataset.classes)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    model = load_model(model_path, num_classes)
    model = model.to(device)
    model.eval()
    
    test_images = get_random_normal_images(dataset_path, num_images=1000)
    
    correct = 0
    results = []

    print("\nTesting 100 random images...")
    print("-" * 50)
    
    for i, (image_path, true_class) in enumerate(test_images, 1):
        predicted_idx = predict_image(model, image_path, transform, device)
        predicted_class = idx_to_class[predicted_idx]
        
        is_correct = (predicted_class == true_class)
        if is_correct:
            correct += 1
            
        results.append({
            'image': os.path.basename(image_path),
            'true_class': true_class,
            'predicted_class': predicted_class,
            'correct': is_correct
        })
        
        print(f"Image {i}/100:")
        print(f"  True Class: {true_class}")
        print(f"  Predicted Class: {predicted_class}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print()

    accuracy = (correct / len(test_images)) * 100
    
    print("=" * 50)
    print("Final Results:")
    print(f"Total Images Tested: {len(test_images)}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()