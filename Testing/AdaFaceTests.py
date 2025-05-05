import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import os,sys
import torch.nn as nn
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Model.Architecture import AdaFaceArchitecture
import matplotlib.pyplot as plt
from Utils.matlab_cp2tform import get_similarity_transform_for_cv2

class AdaFaceAttackFramework:
    def __init__(self, data_dir, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        # Initialize model
        adaface_models = {'ir_50':"E:/AdversarialAttack-2/Model/Weights/adaface_ir50_ms1mv2.ckpt",}
        self.model = AdaFaceArchitecture.build_model("ir_50")
        statedict = torch.load(adaface_models["ir_50"], weights_only=False)['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.landmark = {}
        self.SimScore = []
        with open('Model/lfw_landmark/lfw_landmark.txt') as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n','').split('\t')
            self.landmark[l[0]] = [int(k) for k in l[1:]]
    def alignment(self,src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        crop_size = (112, 112)
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img        
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
    def to_input(self,pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.from_numpy(np.array([brg_img.transpose(2,0,1)])).float().to(self.device)
        return tensor
    def verify_pair(self, img1_path, img2_path, threshold=0.3):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Create landmark keys in format "person/person_0001.jpg"
        is_adv_example = any(suffix in file1 for suffix in ['_fgsm_adv.jpg', '_pgd_adv.jpg', '_bim_adv.jpg', 
                                                        '_mifgsm_adv.jpg', '_cw_adv.jpg', '_spsa_adv.jpg', 
                                                        '_square_adv.jpg'])
        
        # For adversarial examples, use the original file's landmark info
        if is_adv_example:
            base_name = '_'.join(file1.split('_')[:-2]) 
            original_file = base_name + '.jpg'
            landmark_key1 = f"{person1}/{original_file}"
        else:
            landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = self.alignment(img1, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        bgr_tensor_input1 = self.to_input(img1)
        bgr_tensor_input2 = self.to_input(img2)
        with torch.no_grad():
            feature1, _ = self.model(bgr_tensor_input1)
            feature2, _ = self.model(bgr_tensor_input2)
        similarity_score = torch.mm(feature1, feature2.T).item()
        self.SimScore.append(similarity_score)
        return similarity_score > threshold
    def generateFGSMAttack(self, img1_path, img2_path, label=None, epsilon=8/255):
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images using the same approach as verify_pair
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert img2 to tensor using the same preprocessing as in verify_pair
        img2_tensor = self.to_input(img2)
        
        # Create a tensor for img1 that requires gradients
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device).requires_grad_(True)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            # Get features for img2
            feature2, _ = self.model(img2_tensor)
        
        # Get features for img1 (with gradient tracking)
        feature1, _ = self.model(img1_tensor)
                
        # Define loss based on the desired outcome
        if label == 1:  # Same person: attack to decrease similarity
            loss = -torch.mm(feature1, feature2.T)
        else:  # Different person: attack to increase similarity
            loss = torch.mm(feature1, feature2.T)
        
        # Compute gradient using autograd.grad (more efficient than backward)
        grad = torch.autograd.grad(loss, img1_tensor)[0]
        
        # Apply FGSM perturbation (sign of gradient * epsilon)
        perturbed_image = img1_tensor.detach() + epsilon * torch.sign(grad)
        
        # Ensure the perturbed image stays within valid bounds
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
        
        # Convert back to image format
        adv_img = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        adv_img = (adv_img + 0.5) * 255.0
        adv_img = np.clip(adv_img[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_fgsm_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generatePGDAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, steps=20):
        # Set default alpha if not provided
        if alpha is None:
            alpha = epsilon/10
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert img2 to tensor using the same preprocessing as in verify_pair
        img2_tensor = self.to_input(img2)
        
        # Process img1 for attack
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            # Get features for img2
            feature2, _ = self.model(img2_tensor)
        
        # Initialize adversarial example with small random noise
        perturbed_image = img1_tensor.clone().detach()
        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon/2, epsilon/2)
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0).detach()
        
        # Iterative attack
        for _ in range(steps):
            # Set requires_grad
            perturbed_image.requires_grad = True
            
            # Forward pass to get features
            feature_adv, _ = self.model(perturbed_image)
            
            # Calculate similarity score
            similarity = torch.mm(feature_adv, feature2.T)
            
            # Define loss based on the desired outcome
            if label == 1:  # Same person: attack to decrease similarity
                loss = -similarity
            else:  # Different person: attack to increase similarity
                loss = similarity
            
            # Compute gradient using autograd.grad
            grad = torch.autograd.grad(loss, perturbed_image)[0]
            
            # Update and detach adversarial images
            perturbed_image = perturbed_image.detach() + alpha * torch.sign(grad)
            
            # Project back to epsilon ball and valid image range
            delta = torch.clamp(perturbed_image - img1_tensor, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(img1_tensor + delta, min=-1.0, max=1.0).detach()
        
        # Convert back to image format
        adv_img = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        adv_img = (adv_img + 0.5) * 255.0
        adv_img = np.clip(adv_img[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_pgd_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generateBIMAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, iterations=20):
        if alpha is None:
            alpha = epsilon/10
            
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert images to tensors using consistent preprocessing
        img2_tensor = self.to_input(img2)
        
        # Process img1 for attack using the same preprocessing as in to_input
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        ori_img = img1_tensor.clone().detach()
        
        # BIM attack loop
        for _ in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            feature_adv, _ = self.model(adv_img)
            
            # Calculate similarity using matrix multiplication (consistent with verify function)
            similarity = torch.mm(feature_adv, feature2.T)
            
            # Define loss based on attack goal
            if label == 1:  # Same person: attack to decrease similarity
                loss = -similarity
            else:  # Different person: attack to increase similarity
                loss = similarity
            
            # Compute gradients using autograd.grad
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            # Detach from computation graph
            adv_img = adv_img.detach()
            
            # Update adversarial image with sign of gradient (FGSM-like step)
            adv_img = adv_img + alpha * torch.sign(grad)
            
            # Project back into epsilon ball and valid image range
            # Apply BIM-specific clipping approach
            a = torch.clamp(ori_img - epsilon, min=-1.0)
            b = (adv_img >= a).float() * adv_img + (adv_img < a).float() * a
            c = (b > ori_img + epsilon).float() * (ori_img + epsilon) + (b <= ori_img + epsilon).float() * b
            adv_img = torch.clamp(c, min=-1.0, max=1.0).detach()
        
        # Convert back to image format
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_bim_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateMIFGSMAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, iterations=20, decay_factor=0.9):
        if alpha is None:
            alpha = epsilon/10
            
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert images to tensors using consistent preprocessing
        img2_tensor = self.to_input(img2)
        
        # Process img1 using the same preprocessing
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        
        # Initialize the momentum term to zero
        momentum = torch.zeros_like(img1_tensor).to(self.device)
        
        # MI-FGSM attack loop
        for _ in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            feature_adv, _ = self.model(adv_img)
            
            # Calculate similarity using matrix multiplication (consistent with verify function)
            similarity = torch.mm(feature_adv, feature2.T)
            
            # Define loss based on attack goal
            if label == 1:  # Same person: attack to decrease similarity
                loss = -similarity
            else:  # Different person: attack to increase similarity
                loss = similarity
            
            # Compute gradients using autograd.grad
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            # Detach from computation graph
            adv_img = adv_img.detach()
            
            # Normalize gradient (L1 norm)
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / (grad_norm + 1e-10)  # Add small constant to prevent division by zero
            
            # Update momentum term
            momentum = decay_factor * momentum + grad
            
            # Update adversarial image with momentum sign
            adv_img = adv_img + alpha * momentum.sign()
            
            # Project back into epsilon ball and valid image range
            delta = torch.clamp(adv_img - img1_tensor, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(img1_tensor + delta, min=-1.0, max=1.0).detach()
        
        # Convert back to image format
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_mifgsm_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateCWAttack(self, img1_path, img2_path, label=None, c=1.0, kappa=0, steps=30, lr=0.01, threshold=0.3):
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert images to tensors using consistent preprocessing
        img2_tensor = self.to_input(img2)
        
        # Process img1 using the same preprocessing
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        # Functions for CW transformation to tanh space
        def tanh_space(x):
            return 0.5 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x):
            # Ensure values are in valid range for atanh
            x_clamped = torch.clamp(x, min=-0.999, max=0.999)
            return torch.atanh(x_clamped)
        
        # Initialize w in the inverse tanh space
        # We need to map from [-1,1] to [0,1] for tanh space transformation
        img1_norm = (img1_tensor + 1.0) / 2.0
        w = inverse_tanh_space(2 * img1_norm - 1).detach()
        w.requires_grad = True
        
        # Set up optimizer
        optimizer = torch.optim.Adam([w], lr=lr)
        
        # Initialize best adversarial example
        best_adv_images = img1_tensor.clone().detach()
        best_L2 = 1e10 * torch.ones(len(img1_tensor)).to(self.device)
        prev_cost = 1e10
        
        # Prepare loss functions
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        
        # Optimization loop
        for step in range(steps):
            # Get adversarial images in [0,1] space and rescale to original [-1,1] range
            adv_images_norm = tanh_space(w)
            adv_images = adv_images_norm * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
            
            # Calculate L2 distance loss (in normalized space)
            current_L2 = MSELoss(Flatten(adv_images_norm), Flatten(img1_norm)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            # Get features of adversarial image
            feature_adv, _ = self.model(adv_images)
            
            # Calculate similarity using matrix multiplication (consistent with verify function)
            similarity = torch.mm(feature_adv, feature2.T)
            
            # Adapt f-function for similarity based on label
            if label == 1:  # Same person: want to decrease similarity below threshold
                f_loss = torch.clamp(similarity - threshold + kappa, min=0)
            else:  # Different person: want to increase similarity above threshold
                f_loss = torch.clamp(threshold - similarity + kappa, min=0)
            
            # Total cost
            cost = L2_loss + c * f_loss
            
            # Gradient step
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            # Determine which images satisfy the adversarial condition
            if label == 1:  # Same person: success if similarity < threshold
                condition = (similarity < threshold).float()
            else:  # Different person: success if similarity > threshold
                condition = (similarity > threshold).float()
            
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
        
        # Convert back to image format
        perturbed_image = best_adv_images[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        perturbed_image = (perturbed_image + 0.5) * 255.0
        
        # Handle any NaN values that might have been introduced
        perturbed_image = np.nan_to_num(perturbed_image, nan=0.0, posinf=255.0, neginf=0.0)
        
        # Convert back to RGB from BGR and clip to valid range
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_cw_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSPSAAttack(self, img1_path, img2_path, label=None, epsilon=8/255, iterations=100, learning_rate=0.01, delta=0.01):
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert images to tensors using consistent preprocessing
        img2_tensor = self.to_input(img2)
        
        # Process img1 using the same preprocessing
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        
        # Function to compute loss based on goal
        def compute_loss(perturbed_img):
            with torch.no_grad():
                feature_perturbed, _ = self.model(perturbed_img)
                # Calculate similarity using matrix multiplication (consistent with verify function)
                similarity = torch.mm(feature_perturbed, feature2.T)
                
                if label == 1:  # Same person: want to decrease similarity (below threshold)
                    return similarity  # Maximize this loss to decrease similarity
                else:  # Different person: want to increase similarity (above threshold)
                    return -similarity  # Minimize this loss to increase similarity
        
        # SPSA attack loop
        for i in range(iterations):
            # Create random perturbation direction (Bernoulli distribution {-1, 1})
            bernoulli = torch.randint(0, 2, adv_img.shape).to(self.device) * 2 - 1  # -1 or 1
            
            # Evaluate loss at points in both positive and negative directions
            pos_perturbed = adv_img + delta * bernoulli
            pos_perturbed = torch.clamp(pos_perturbed, -1.0, 1.0)
            loss_pos = compute_loss(pos_perturbed)
            
            neg_perturbed = adv_img - delta * bernoulli
            neg_perturbed = torch.clamp(neg_perturbed, -1.0, 1.0)
            loss_neg = compute_loss(neg_perturbed)
            
            # Estimate gradient using finite differences
            gradient_estimate = (loss_pos - loss_neg) / (2 * delta)
            
            # Apply estimated gradient to update the adversarial example
            # Note: we want to minimize the loss, so we use negative gradient direction
            adv_img = adv_img - learning_rate * gradient_estimate * bernoulli
            
            # Project back to epsilon-ball around original image and ensure valid pixel range
            delta_img = torch.clamp(adv_img - img1_tensor, min=-epsilon, max=epsilon)
            adv_img = img1_tensor + delta_img
            adv_img = torch.clamp(adv_img, -1.0, 1.0)
            
            # Optionally reduce learning rate over time (learning rate decay)
            learning_rate = learning_rate * 0.99
        
        # Convert back to image format
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_spsa_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSquareAttack(self, img1_path, img2_path, label=None, n_iters=1000, p_init=0.1, epsilon=8/255, threshold=0.3):
        # Extract person and file information from paths
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        # Create landmark keys
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load and align images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        # Convert images to tensors using consistent preprocessing
        img2_tensor = self.to_input(img2)
        
        # Process img1 using the same preprocessing
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        # Calculate target features with no gradient tracking
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        # Initialize adversarial example with the original image
        x_adv = img1_tensor.clone().detach()
        
        # Function to evaluate similarity using matrix multiplication
        def compute_similarity(x):
            with torch.no_grad():
                feature_x, _ = self.model(x)
                similarity = torch.mm(feature_x, feature2.T)
                return similarity.item()
        
        # Determine the attack goal based on the label
        if label == 1:  # Same person pair - want to decrease similarity
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim < curr_sim
        else:  # Different person pair - want to increase similarity
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim > curr_sim
        
        # Get image dimensions
        _, _, h, w = img1_tensor.shape  # Should be (1, 3, h, w)
        
        # Main attack loop
        for i in range(n_iters):
            # Calculate current p (decreases over iterations)
            p = p_init * (1 - i / n_iters)**0.5
            
            # Calculate square size based on p
            s = int(round(np.sqrt(p * h * w)))
            s = max(1, min(s, h, w))
            
            # Randomly select square position
            h_start = np.random.randint(0, h - s + 1)
            w_start = np.random.randint(0, w - s + 1)
            
            # Randomly select channel to perturb (or all channels)
            channel = np.random.choice([-1, 0, 1, 2])  # -1 means all channels
            
            # Create a copy of the current best adversarial example
            x_new = x_adv.clone().detach()
            
            # Generate perturbation in normalized space [-1, 1]
            if channel == -1:  # Apply to all channels
                noise = torch.empty((1, 3, s, s), device=self.device).uniform_(-2*epsilon, 2*epsilon)
                x_new[0, :, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, :, h_start:h_start+s, w_start:w_start+s] + noise[0], -1.0, 1.0)
            else:  # Apply to specific channel
                noise = torch.empty((1, s, s), device=self.device).uniform_(-2*epsilon, 2*epsilon)
                x_new[0, channel, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, channel, h_start:h_start+s, w_start:w_start+s] + noise[0], -1.0, 1.0)
            
            # Evaluate the new example
            new_similarity = compute_similarity(x_new)
            
            # Update if better
            if is_better(new_similarity, best_similarity):
                x_adv = x_new
                best_similarity = new_similarity
            
            # Early stopping if we've reached a very good solution
            if (label == 1 and best_similarity < threshold) or (label == 0 and best_similarity > threshold):
                break
        
        # Convert back to image format
        perturbed_image = x_adv[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Reverse the normalization: (-1,1) → (0,1) → (0,255)
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  # Convert back to RGB and clip
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_square_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def evaluate_attack(self, attack_type):
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            try:                
                # Apply attack
                if attack_type == "FGSM":
                    adv_img_path = self.generateFGSMAttack(img1_path, img2_path, label)
                elif attack_type == "PGD": 
                    adv_img_path = self.generatePGDAttack(img1_path, img2_path, label)
                elif attack_type == "BIM":
                    adv_img_path = self.generateBIMAttack(img1_path, img2_path, label)
                elif attack_type == "MIFGSM":
                    adv_img_path = self.generateMIFGSMAttack(img1_path, img2_path, label)
                elif attack_type == "CW":
                    adv_img_path = self.generateCWAttack(img1_path, img2_path, label)

                elif attack_type == "SPSA":
                    adv_img_path = self.generateSPSAAttack(img1_path, img2_path, label)
                elif attack_type == "Square":
                    adv_img_path = self.generateSquareAttack(img1_path, img2_path, label)
                # Verify
                prediction = self.verify_pair(adv_img_path, img2_path)
                
                # Update results
                if label == 1:
                    if prediction: 
                        results['true_positive'] += 1
                    else: 
                        results['false_negative'] += 1
                else:
                    if prediction: 
                        results['false_positive'] += 1
                    else: 
                        results['true_negative'] += 1
                
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

        for attack_type in ["FGSM", "PGD", "BIM", "MIFGSM", "CW", "SPSA", "Square"]:
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
        self.save_l2_to_txt()

        return results
    def save_l2_to_txt(self, filename="L2_Values/SimScore_values_Ada.txt"):
        with open(filename, 'w') as f:
            for value in self.SimScore:
                f.write(f"{value}\n")
        print(f"Saved {len(self.SimScore)} Similarity Score values to {filename}")


if __name__ == "__main__":
    framework = AdaFaceAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
    )
    results = framework.run_evaluation()
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
