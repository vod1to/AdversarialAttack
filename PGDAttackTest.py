import os
import random
import torch
import torch.nn as nn
import torchattacks
from torchvision import datasets, transforms, models
from PIL import Image


def load_model(model_path, num_classes):
    """Load the trained model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def get_random_normal_images(dataset_path, num_images):
    """Retrieve random image paths and class names (labels) from the dataset directory"""
    all_images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.basename(root)
                image_path = os.path.join(root, file)
                all_images.append((image_path, class_name))
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    return selected_images

def predict_image(model, image_tensor, device):
    """Make prediction for a single image tensor"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def attack_image(selected_images, model, device, transform, class_to_idx):
    """Apply PGD attack to images"""
    adv_images = []
    atk = torchattacks.PGD(model=model, eps=8/255, alpha=2/255, steps=10, random_start=False)
    
    for image_path, label in selected_images:
        # Convert the label to its numerical class index
        label_idx = class_to_idx[label]
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate adversarial example
        adv_image = atk(image_tensor, torch.tensor([label_idx]).to(device))
        adv_images.append((adv_image, label_idx))  # Append both the adversarial image tensor and its label index
    
    return adv_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model_path = "outputs/ResNet18Trained.pth" 
    dataset_path = "E:/lfw/lfw-py/lfw_funneled"  
    
    # Load dataset to get class information
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    num_classes = len(dataset.classes)
    class_to_idx = dataset.class_to_idx  # Map from class names to numerical indices
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse map for predictions

    # Load and prepare model
    model = load_model(model_path, num_classes)
    model = model.to(device)
    model.eval()
    
    # Select and attack images
    images = get_random_normal_images(dataset_path, num_images=1000)
    test_images = attack_image(selected_images=images, model=model, device=device, transform=transform, class_to_idx=class_to_idx)
    
    correct = 0
    results = []

    print("\nTesting 1000 adversarial images...")
    print("-" * 50)
    
    for i, (adv_image, true_class_idx) in enumerate(test_images, 1):
        # Get prediction on adversarial image
        predicted_idx = predict_image(model, adv_image, device)
        predicted_class = idx_to_class[predicted_idx]
        true_class = idx_to_class[true_class_idx]
        
        is_correct = (predicted_idx == true_class_idx)
        if is_correct:
            correct += 1
            
        results.append({
            'image': os.path.basename(images[i-1][0]),  # Original image name for reference
            'true_class': true_class,
            'predicted_class': predicted_class,
            'correct': is_correct
        })
        
        print(f"Image {i}/100:")
        print(f"  True Class: {true_class}")
        print(f"  Predicted Class: {predicted_class}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print()

    # Calculate and print accuracy
    accuracy = (correct / len(test_images)) * 100
    
    print("=" * 50)
    print("Final Results:")
    print(f"Total Images Tested: {len(test_images)}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
