import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import os,sys
import torch.nn as nn
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

    def verify_pair(self, img1_path, img2_path, threshold=1.2):
        # Read and preprocess images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        # Convert to torch tensor and normalize
        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        
        # VGG Face mean subtraction
        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        self.model.eval()      
        with torch.no_grad():
            # Forward pass until fc7 for both images
            feat1 = self.model.get_features(img1)
            feat2 = self.model.get_features(img2)
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            # Compute L2 distance
            l2_distance = torch.norm(feat1 - feat2, p=2).item()
            return l2_distance < threshold
    def generateFGSMAttack(self, img1_path, img2_path, label = None):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)

        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        img1_adv = img1.clone().detach().requires_grad_(True)
        epsilon = 8/255  # Attack strength parameter
    
        self.model.eval()
        feat1 = self.model.get_features(img1_adv)
        feat2 = self.model.get_features(img2)
    
        # Normalize features
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Compute L2 distance
        distance = torch.norm(feat1 - feat2, p=2)
        
        # Define loss based on attack goal
        if label == 1:  # We want to decrease distance (make different people look same)
            loss = distance
        else:  # We want to increase distance (make same person look different)
            loss = -distance
                
        # Create adversarial example with FGSM
        with torch.no_grad():
            # Get the sign of the gradients
            grad_sign = torch.autograd.grad(loss, img1_adv)[0]
            # Apply perturbation
            perturbed_image = img1 + epsilon * grad_sign.sign()
            # Clamp to ensure valid pixel range
            perturbed_image = torch.clamp(perturbed_image, 0, 255)
            
        
        # Convert back to image format and save
        adv_img = perturbed_image[0].permute(1, 2, 0).cpu().numpy()
        adv_img += np.array([129.1863, 104.7624, 93.5940]) # Add back the mean
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_fgsm_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generatePGDAttack(self, img1_path, img2_path, label = None):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)

        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        epsilon = 8/255  # Total perturbation constraint
        alpha = epsilon/10  # Step size
        steps = 20  # Number of attack iterations

        self.model.eval()
        perturbed_image = img1.clone().detach()
        
        # Add small random noise to start
        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 255).detach()
        
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)


        # Iterative attack
        for _ in range(steps):
            # Set requires_grad
            perturbed_image.requires_grad = True
            
            # Forward pass to get features
            feat1 = self.model.get_features(perturbed_image)
            
            # Normalize features
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            # Compute L2 distance
            distance = torch.norm(feat1 - feat2, p=2)
            
            # Define loss based on attack goal
            if label == 1:  # We want to decrease distance (make different people look same)
                loss = distance
            else:  # We want to increase distance (make same person look different)
                loss = -distance
            

            
            # Take gradient step
            grad = torch.autograd.grad(loss, perturbed_image)[0]
        
            # Update and detach adversarial images
            perturbed_image = perturbed_image.detach() - alpha * grad.sign()  # Note the minus sign
            
            # Project back to epsilon ball and valid image range
            delta = torch.clamp(perturbed_image - img1, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(img1 + delta, min=0, max=255).detach()
        
        # Convert to image and save
        adv_img = perturbed_image[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_img += np.array([129.1863, 104.7624, 93.5940])
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_pgd_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generateBIMAttack(self, img1_path, img2_path, label=None):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)

        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # BIM parameters
        epsilon = 8/255      # Total perturbation constraint
        alpha = epsilon/10   # Step size per iteration
        iterations = 20      # Number of attack iterations

        # Extract features from target image
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Initialize adversarial example with the original image
        adv_img = img1.clone().detach()
        ori_img = img1.clone().detach()
        
        # BIM attack loop
        for i in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            feat1 = self.model.get_features(adv_img)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            # Compute distance between feature vectors
            distance = torch.norm(feat1 - feat2, p=2)
            
            # Define loss based on attack goal
            if label == 1:  # Decrease distance (make different people look same)
                loss = distance
            else:  # Increase distance (make same person look different)
                loss = -distance
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            # Detach from computation graph
            adv_img = adv_img.detach()
            
            # Update adversarial image with sign of gradient (FGSM-like step)
            adv_img = adv_img - alpha * torch.sign(grad)
            a = torch.clamp(ori_img - epsilon, min=0)
            b = (adv_img >= a).float() * adv_img + (adv_img < a).float() * a
            c = (b > ori_img + epsilon).float() * (ori_img + epsilon) + (b <= ori_img + epsilon).float() * b
            
            # Ensure pixel values stay within valid range
            adv_img = torch.clamp(c, min=0, max=255).detach()
        
        # Convert to image and save
        adv_output = adv_img[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_output += np.array([129.1863, 104.7624, 93.5940])
        adv_output = np.clip(adv_output, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_bim_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        
        return output_path

    def generateMIFGSMAttack(self, img1_path, img2_path, label=None):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)

        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # MI-FGSM parameters
        epsilon = 8/255       # Total perturbation constraint
        alpha = epsilon/10    # Step size per iteration
        iterations = 20       # Number of attack iterations
        decay_factor = 0.9    # Momentum decay factor

        # Extract features from target image
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Initialize adversarial example with the original image
        adv_img = img1.clone().detach()
        
        # Initialize the momentum term to zero
        momentum = torch.zeros_like(img1).to(self.device)
        
        # MI-FGSM attack loop
        for i in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            feat1 = self.model.get_features(adv_img)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            # Compute distance between feature vectors
            distance = torch.norm(feat1 - feat2, p=2)
            
            # Define loss based on attack goal
            if label == 1:  # Decrease distance (make different people look same)
                loss = distance
            else:  # Increase distance (make same person look different)
                loss = -distance
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            # Detach from computation graph
            adv_img = adv_img.detach()
            
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / grad_norm 
            
            # Update momentum term
            grad = grad + momentum * decay_factor
            momentum = grad            
            adv_img = adv_img - alpha * grad.sign()
            
            delta = torch.clamp(adv_img - img1, min=-epsilon, max=epsilon)
            adv_img = img1 + delta
            adv_img = torch.clamp(adv_img, min=0, max=255)
        
        # Convert to image and save
        adv_output = adv_img[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_output += np.array([129.1863, 104.7624, 93.5940])
        adv_output = np.clip(adv_output, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_mifgsm_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        
        return output_path
    def generateCWAttack(self, img1_path, img2_path, label=None, c=1.0, kappa=0, steps=30, lr=0.01):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        img1 = torch.Tensor(img1).float().permute(2, 0, 1).reshape(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).reshape(1, 3, 224, 224)

        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().reshape(1, 3, 1, 1)
        img1 -= mean
        img2 -= mean

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Following torchattacks approach
        self.model.eval()
        
        # Functions from torchattacks
        def tanh_space(x):
            return 0.5 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x):
            # Inverse of tanh_space in the range [0, 1]
            return 0.5 * torch.log((1 + x*2 - 1) / (1 - (x*2 - 1)))
        
        # Initialize w in the inverse tanh space
        w = inverse_tanh_space(img1 / 255.0).detach()  # Convert to [0,1] range first
        w.requires_grad = True
        
        # Set up optimizer
        optimizer = torch.optim.Adam([w], lr=lr)
        
        # Initialize best adversarial example
        best_adv_images = img1.clone().detach()
        best_L2 = 1e10 * torch.ones((len(img1))).to(self.device)
        prev_cost = 1e10
        
        # Extract features from target image
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Prepare loss functions
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        
        # Optimization loop
        for step in range(steps):
            # Get adversarial images in [0,1] space and rescale to original range
            adv_images_norm = tanh_space(w)
            adv_images = adv_images_norm * 255.0  # Back to [0,255] range
            
            # Calculate L2 distance loss (in pixel space)
            current_L2 = MSELoss(Flatten(adv_images_norm), Flatten(img1 / 255.0)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            # Get features of adversarial image
            feat1 = self.model.get_features(adv_images)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            # Calculate feature distance
            distance = torch.norm(feat1 - feat2, p=2, dim=1)
            threshold = 1.0

            
            # Adapt f-function from torchattacks for our face recognition task
            if label == 1:  # Decrease distance (make different people look same)
                # We want distance to be minimized, so penalize if it's large
                f_loss = torch.clamp(distance - kappa, min=0).sum()
            else:  # Increase distance (make same person look different)
                # We want distance to be maximized, so penalize if it's small
                # Assuming a threshold of 1.0 for simplicity
                f_loss = torch.clamp(threshold - distance + kappa, min=0).sum()
            
            # Total cost
            cost = L2_loss + c * f_loss
            
            # Gradient step
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            # Update best adversarial images
            # For face recognition, success condition is based on distance threshold
            if label == 1:  # We want small distance
                condition = (distance < threshold).float()
            else:  # We want large distance
                condition = (distance > threshold).float()
            
            # Filter out images that either don't meet the condition or have larger L2
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            
            # Update best adversarial images
            mask = mask.view([-1] + [1] * (len(adv_images.shape) - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            
            # Early stop when loss does not converge
            if step % max(steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    break
                prev_cost = cost.item()
        
        # Add mean back to final result
        adv_output = best_adv_images[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_output += np.array([129.1863, 104.7624, 93.5940])

        adv_output = np.nan_to_num(adv_output, nan=0.0, posinf=255.0, neginf=0.0)
        adv_output = np.clip(adv_output, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_cw_torchattacks_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        
        return output_path
    def evaluate_attack(self, attack_type):
        """Evaluate attack performance"""
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            try:                
                # Apply attack
                if attack_type == "FGSM":
                    adv_img_path = self.generateFGSMAttack(img1_path, img2_path, label)
                elif attack_type == "PGD": # PGD
                    adv_img_path = self.generatePGDAttack(img1_path, img2_path, label)
                elif attack_type == "BIM":
                    adv_img_path = self.generateBIMAttack(img1_path, img2_path, label)
                elif attack_type == "MIFGSM":
                    adv_img_path = self.generateMIFGSMAttack(img1_path, img2_path, label)
                elif attack_type == "CW":
                    adv_img_path = self.generateCWAttack(img1_path, img2_path, label)
                # Verify
                prediction = self.verify_pair(adv_img_path, img2_path)
                
                # Update results
                if label == 1:
                    if prediction: 
                        results['true_positive'] += 1
                        print(f"True Positive: {img1_path} - {img2_path}")
                    else: 
                        results['false_negative'] += 1
                        print(f"False Negative: {img1_path} - {img2_path}")
                else:
                    if prediction: 
                        results['false_positive'] += 1
                        print(f"False Positive: {img1_path} - {img2_path}")
                    else: 
                        results['true_negative'] += 1
                        print(f"True Negative: {img1_path} - {img2_path}")
                
                # Clean up
                if os.path.exists(adv_img_path):
                    os.remove(adv_img_path)
                
            except Exception as e:
                print(f"Error processing pair: {e}")
        
        # Calculate metrics
        total = sum(results.values())
        
        if total == 0:
            return {'accuracy': 0, 'far': 0, 'frr': 0, 'attack_success_rate': 0}        
        return {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] /  (results['false_positive'] + results['true_negative']),
            'frr': results['false_negative'] / (results['false_negative'] + results['true_positive']),
            'attack_success_rate': (results['false_negative'] + results['false_positive']) / total}
    def run_evaluation(self):
        results = {}
        # Attack evaluations
        for attack_type in ["CW"]:
            print(f"\nEvaluating {attack_type} attack...")
            results[attack_type] = self.evaluate_attack(attack_type)
        
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
    print(framework.cw_test())