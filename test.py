import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_model(model_path, num_classes):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def get_random_images(dataset_path, num_images=100):
    """Get random images from the dataset"""
    all_images = []
    class_names = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.basename(root)
                image_path = os.path.join(root, file)
                all_images.append((image_path, class_name))
    
    # Randomly select images
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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Paths
    dataset_path = "E:/lfw/lfw-py/lfw_funneled"  # Replace with your LFW dataset path
    model_path = "outputs/face_recognition_model.pth"  # Path to your saved model
    
    # Load dataset to get class information
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    num_classes = len(dataset.classes)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    # Load model
    model = load_model(model_path, num_classes)
    model = model.to(device)
    model.eval()
    
    # Get random images
    test_images = get_random_images(dataset_path, num_images=100)
    
    # Test the model
    correct = 0
    results = []

    print("\nTesting 100 random images...")
    print("-" * 50)
    
    for i, (image_path, true_class) in enumerate(test_images, 1):
        # Get prediction
        predicted_idx = predict_image(model, image_path, transform, device)
        predicted_class = idx_to_class[predicted_idx]
        
        # Check if prediction is correct
        is_correct = (predicted_class == true_class)
        if is_correct:
            correct += 1
            
        # Store results
        results.append({
            'image': os.path.basename(image_path),
            'true_class': true_class,
            'predicted_class': predicted_class,
            'correct': is_correct
        })
        
        # Print progress
        print(f"Image {i}/100:")
        print(f"  True Class: {true_class}")
        print(f"  Predicted Class: {predicted_class}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print()

    # Calculate and print final statistics
    accuracy = (correct / len(test_images)) * 100
    
    print("=" * 50)
    print("Final Results:")
    print(f"Total Images Tested: {len(test_images)}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
if __name__ == "__main__":
    main()