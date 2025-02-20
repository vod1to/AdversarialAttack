import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Model.Architecture.VGGFaceArchitecture import VGGFace

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
            if len(pairs) == 50:
                break
        
        # Different person pairs
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
            if len(pairs) == 100:
                break
        return pairs
    def extract_features(self, img_path):
        """
        Extract features from an image using multi-scale and multiple crops as per paper:
        - 3 scales (256, 384, 512)
        - 5 crops per scale (corners + center)
        - Horizontal flips for each crop
        Returns normalized feature vector
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        features_list = []
        scales = [256, 384, 512]
        
        for scale in scales:
            # Resize image preserving aspect ratio
            height, width = img.shape[:2]
            scale_factor = scale / float(min(height, width))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_img = cv2.resize(img, (new_width, new_height))
            
            # Convert to torch tensor
            img_tensor = torch.Tensor(resized_img).float().permute(2, 0, 1)
            
            # Get crops
            crops = []
            h, w = img_tensor.shape[1:]
            
            # Center crop
            crop_size = 224
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            center_crop = img_tensor[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
            crops.append(center_crop)
            
            # Corner crops
            corner_crops = [
                img_tensor[:, :crop_size, :crop_size],  # top-left
                img_tensor[:, :crop_size, -crop_size:],  # top-right
                img_tensor[:, -crop_size:, :crop_size],  # bottom-left
                img_tensor[:, -crop_size:, -crop_size:]  # bottom-right
            ]
            crops.extend(corner_crops)
            
            # Add flipped versions
            flipped_crops = [torch.flip(crop, [2]) for crop in crops]
            crops.extend(flipped_crops)
            
            # Process each crop
            for crop in crops:
                crop = crop.view(1, 3, 224, 224)
                
                # Subtract mean (as mentioned in paper Section 4.3)
                mean = torch.Tensor([129.1863, 104.7624, 93.5940]).float().view(1, 3, 1, 1)
                crop = crop - mean
                
                # Move to device
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    crop = crop.cuda()
                
                # Extract features
                with torch.no_grad():
                    feat = self.model.get_features(crop)
                    features_list.append(feat)
        
        # Average all features
        final_features = torch.mean(torch.cat(features_list, 0), dim=0, keepdim=True)
        
        # L2 normalize
        final_features = F.normalize(final_features, p=2, dim=1)
        
        return final_features

    def verify_pair(self, img1_path, img2_path, threshold=1.2):
        """
        Verify if two face images belong to the same person.
        Uses multi-scale feature extraction and normalized L2 distance.
        """
        try:
            # Extract features with multi-scale and multiple crops
            feat1 = self.extract_features(img1_path)
            feat2 = self.extract_features(img2_path)
            
            # Compute L2 distance between normalized features
            l2_distance = torch.norm(feat1 - feat2, p=2).item()
            
            # For normalized vectors:
            # - Distance of 0 means identical
            # - Distance of 2 means opposite
            return l2_distance < threshold
            
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            return False, float('inf')
    def run_evaluation(self):
        results = {}
        # Clean performance
        print("Evaluating clean performance...")
        clean_results = {'true_positive': 0, 'true_negative': 0,
                        'false_positive': 0, 'false_negative': 0}
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            prediction = self.verify_pair(img1_path, img2_path)
            if label == 1:
                if prediction: clean_results['true_positive'] += 1
                else: clean_results['false_negative'] += 1
            else:
                if prediction: clean_results['false_positive'] += 1
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
        model_path='E:/AdversarialAttack-2/Model/Weights/vgg_face_dag.pth'
    )
    results = framework.run_evaluation()
    
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")