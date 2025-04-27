import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import os,sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Model.Architecture.SphereFaceArchitecture import sphere20a
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from torch.autograd import Variable
from sphereUtils.matlab_cp2tform import get_similarity_transform_for_cv2
class SphereAttackFramework:
    def __init__(self, data_dir, model_path, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        # Initialize model
        self.model = sphere20a().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.landmark = {}
        with open('Model/lfw_landmark/lfw_landmark.txt') as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n','').split('\t')
            self.landmark[l[0]] = [int(k) for k in l[1:]]
    def alignment(self,src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        crop_size = (96, 112)
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
            if len(pairs) ==50:
                break
        # Different person pairs
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
            if len(pairs) ==100:
                break
        return pairs
    def verify_pair(self, img1_path, img2_path, threshold=0.35):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        is_adv_example = any(suffix in file1 for suffix in ['_fgsm_adv.jpg', '_pgd_adv.jpg', '_bim_adv.jpg', 
                                                        '_mifgsm_adv.jpg', '_cw_adv.jpg', '_spsa_adv.jpg', 
                                                        '_square_adv.jpg'])
        
        # For adversarial examples, use the original file's landmark info
        if is_adv_example:

            base_name = '_'.join(file1.split('_')[:-2]) 
            original_file = base_name + '.jpg'
            landmark_key1 = f"{person1}/{original_file}"
            print("FOUND")
        else:
            landmark_key1 = f"{person1}/{file1}"
            
        landmark_key2 = f"{person2}/{file2}"
        
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = self.alignment(img1, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])

        
        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.0
        
        img = np.vstack(imglist)
                
        # Extract features
        with torch.no_grad():
            img = Variable(torch.from_numpy(img).float()).to(self.device)
            f,_ = self.model(img)

        f1,f2 = f[0],f[2]
        cosine_similarity = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
        # Compute cosine similarity
        return cosine_similarity > threshold
    def generateFGSMAttack(self, img1_path, img2_path, label=None):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # Process images for feature extraction
        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.0
        
        img = np.vstack(imglist)
        
        # Create adversarial version of the first image that requires gradients
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        img1_adv = torch.from_numpy(img1_processed).float().to(self.device).requires_grad_(True)
        
        # Extract features for similarity calculation
        with torch.no_grad():
            img_tensor = torch.from_numpy(img).float().to(self.device)
            f, _ = self.model(img_tensor)
        
        f1, f2 = f[0], f[2]
        
        # Now run the model on the adversarial image to compute gradients
        f_adv, _ = self.model(img1_adv)
        cosine_similarity = torch.dot(f_adv[0], f2)/(torch.norm(f_adv[0])*torch.norm(f2)+1e-5)

        if label == 1:  
            loss = -cosine_similarity  
        else:  
            loss = cosine_similarity  
        
        # Calculate gradient
        grad_sign = torch.autograd.grad(loss, img1_adv)[0]

        
        epsilon = 8/255  # Attack strength parameter
        
        # Apply perturbation in the normalized space
        perturbed_image = img1_adv.detach() + epsilon * grad_sign
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)  # Clamp to valid normalized range
        
        # Convert back to image format (0-255 range)
        perturbed_image = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_fgsm_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generatePGDAttack(self, img1_path, img2_path, label=None):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # PGD attack parameters
        epsilon = 8/255  # Total perturbation constraint
        alpha = epsilon/10  # Step size
        steps = 20  # Number of attack iterations
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
        # Initialize adversarial example with small random noise
        perturbed_image = img1_tensor.clone().detach()
        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon/2, epsilon/2)
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0).detach()
        
        # Iterative attack
        for _ in range(steps):
            # Set requires_grad
            perturbed_image.requires_grad = True
            
            # Forward pass to get features
            f_adv, _ = self.model(perturbed_image)
            
            # Calculate cosine similarity (same as in FGSM function)
            cosine_similarity = torch.dot(f_adv[0], f2)/(torch.norm(f_adv[0])*torch.norm(f2)+1e-5)
            
            if label == 1:
                loss = -cosine_similarity
            else:  
                loss = cosine_similarity
            
            # Take gradient step
            grad = torch.autograd.grad(loss, perturbed_image)[0]
            
            # Update and detach adversarial images
            perturbed_image = perturbed_image.detach() + alpha * grad.sign() 
            
            # Project back to epsilon ball and valid image range
            delta = torch.clamp(perturbed_image - img1_tensor, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(img1_tensor + delta, min=-1.0, max=1.0).detach()
        
        # Convert back to image format (0-255 range)
        perturbed_image = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_pgd_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateBIMAttack(self, img1_path, img2_path, label=None):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # BIM parameters
        epsilon = 8/255      # Total perturbation constraint
        alpha = epsilon/10   # Step size per iteration
        iterations = 20      # Number of attack iterations
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        ori_img = img1_tensor.clone().detach()
        
        # BIM attack loop
        for i in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            f_adv, _ = self.model(adv_img)
            
            # Calculate cosine similarity (same as in FGSM function)
            cosine_similarity = torch.dot(f_adv[0], f2)/(torch.norm(f_adv[0])*torch.norm(f2)+1e-5)
            
            # Define loss based on attack goal
            if label == 1:  # We want to increase similarity (make different people look same)
                loss = -cosine_similarity
            else:  # We want to decrease similarity (make same person look different)
                loss = cosine_similarity
            
            # Compute gradients
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
        
        # Convert back to image format (0-255 range)
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_bim_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateMIFGSMAttack(self, img1_path, img2_path, label=None):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # MI-FGSM parameters
        epsilon = 8/255       # Total perturbation constraint
        alpha = epsilon/10    # Step size per iteration
        iterations = 20       # Number of attack iterations
        decay_factor = 0.9    # Momentum decay factor
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        
        # Initialize the momentum term to zero
        momentum = torch.zeros_like(img1_tensor).to(self.device)
        
        # MI-FGSM attack loop
        for i in range(iterations):
            # Reset gradients
            adv_img.requires_grad = True
            
            # Forward pass to get features
            f_adv, _ = self.model(adv_img)
            
            # Calculate cosine similarity 
            cosine_similarity = torch.dot(f_adv[0], f2)/(torch.norm(f_adv[0])*torch.norm(f2)+1e-5)
            
            # Define loss based on attack goal
            if label == 1:  # We want to increase similarity (make different people look same)
                loss = -cosine_similarity
            else:  # We want to decrease similarity (make same person look different)
                loss = cosine_similarity
            
            # Compute gradients
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
        
        # Convert back to image format (0-255 range)
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_mifgsm_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateCWAttack(self, img1_path, img2_path, label=None, c=1.0, kappa=0, steps=30, lr=0.01):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Functions for CW transformation to tanh space
        def tanh_space(x):
            return 0.5 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x):
            # Ensure values are in valid range for atanh
            x_clamped = torch.clamp(x * 2 - 1, min=-0.999, max=0.999)
            return torch.atanh(x_clamped)
        
        # Initialize w in the inverse tanh space
        # First normalize img1_tensor to [0,1] range before inverse tanh
        img1_norm = (img1_tensor + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        w = inverse_tanh_space(img1_norm).detach()
        w.requires_grad = True
        
        # Set up optimizer
        optimizer = torch.optim.Adam([w], lr=lr)
        
        # Initialize best adversarial example
        best_adv_images = img1_tensor.clone().detach()
        best_L2 = 1e10 * torch.ones((len(img1_tensor))).to(self.device)
        prev_cost = 1e10
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
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
            f_adv, _ = self.model(adv_images)
            
            # Calculate cosine similarity (same as in FGSM function)
            cosine_similarity = torch.dot(f_adv[0], f2)/(torch.norm(f_adv[0])*torch.norm(f2)+1e-5)
            
            # Adapt f-function for cosine similarity
            threshold = 0.35  # Threshold for cosine similarity
            
            if label == 1: 
                f_loss = torch.clamp(cosine_similarity - threshold + kappa, min=0)
            else:  
                f_loss = torch.clamp(threshold - cosine_similarity + kappa, min=0)
            
            # Total cost
            cost = L2_loss + c * f_loss
            
            # Gradient step
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            

            if label == 1: 
                condition = (cosine_similarity < threshold).float()
            else:  
                condition = (cosine_similarity > threshold).float()
            
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
        
        # Convert back to image format (0-255 range)
        perturbed_image = best_adv_images[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.nan_to_num(perturbed_image, nan=0.0, posinf=255.0, neginf=0.0)
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_cw_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSPSAAttack(self, img1_path, img2_path, label=None):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # SPSA parameters
        epsilon = 8/255        # Total perturbation constraint
        iterations = 100       # Number of attack iterations
        learning_rate = 0.01   # Learning rate for optimization
        delta = 0.01           # Perturbation size for gradient estimation
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
        # Initialize adversarial example with the original image
        adv_img = img1_tensor.clone().detach()
        
        # Function to compute loss based on goal
        def compute_loss(perturbed_img):
            with torch.no_grad():
                f_perturbed, _ = self.model(perturbed_img)
                cosine_similarity = torch.dot(f_perturbed[0], f2)/(torch.norm(f_perturbed[0])*torch.norm(f2)+1e-5)
                
                # Note: For label=1 (same person pair), we want to decrease similarity
                # For label=0 (different person pair), we want to increase similarity
                # This is the opposite of what the original code indicates
                if label == 1:  # Same person pair - want to decrease similarity
                    return cosine_similarity 
                else:  # Different person pair - want to increase similarity
                    return -cosine_similarity
        
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
        
        # Convert back to image format (0-255 range)
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
        # Save the adversarial example
        output_path = img1_path.replace('.jpg', '_spsa_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSquareAttack(self, img1_path, img2_path, label=None, n_iters=1000, p_init=0.1):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  # Get the person's name (folder name)
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    # Get the filename
        file2 = img2_parts[-1]
        
        # Create landmark keys in format "person/person_0001.jpg"
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Try to get landmarks and apply alignment
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2, self.landmark[landmark_key2])
        
        # Process images for feature extraction
        img1_processed = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img1_processed = (img1_processed - 127.5) / 128.0
        
        img2_processed = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        img2_processed = (img2_processed - 127.5) / 128.0
        
        img1_tensor = torch.from_numpy(img1_processed).float().to(self.device)
        img2_tensor = torch.from_numpy(img2_processed).float().to(self.device)
        
        # Square attack parameters
        epsilon = 8/255
        
        # Get features for the second image (target)
        with torch.no_grad():
            f2, _ = self.model(torch.cat([img2_tensor, torch.flip(img2_tensor, [3])]))
            f2 = f2[0]  # Use only the direct view, not the flipped one
        
        # Initialize adversarial example with the original image
        x_adv = img1_tensor.clone().detach()
        
        # Function to evaluate similarity
        def compute_similarity(x):
            with torch.no_grad():
                f_x, _ = self.model(x)
                cosine_similarity = torch.dot(f_x[0], f2)/(torch.norm(f_x[0])*torch.norm(f2)+1e-5)
                return cosine_similarity.item()
        
        # Determine the best similarity based on the attack goal
        if label == 1:  # Same person pair - want to decrease similarity
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim < curr_sim
        else:  # Different person pair - want to increase similarity
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim > curr_sim
        
        # Reshape image for easier manipulation
        h, w = 112, 96  # Image height and width for your model
        
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
            noise = torch.empty((1, 1 if channel != -1 else 3, s, s), device=self.device).uniform_(-epsilon, epsilon)
            
            # Apply the perturbation to the selected region
            if channel == -1:  # Apply to all channels
                x_new[0, :, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, :, h_start:h_start+s, w_start:w_start+s] + noise, -1.0, 1.0)
            else:  # Apply to specific channel
                x_new[0, channel, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, channel, h_start:h_start+s, w_start:w_start+s] + noise.squeeze(1), -1.0, 1.0)
            
            new_similarity = compute_similarity(x_new)
            
            if is_better(new_similarity, best_similarity):
                x_adv = x_new
                best_similarity = new_similarity
                
            # Early stopping if we've reached a very good solution
            threshold = 0.35  # Threshold for similarity
            if (label == 1 and best_similarity < threshold) or (label == 0 and best_similarity > threshold):
                break
        
        # Convert back to image format (0-255 range)
        perturbed_image = x_adv[0].permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 128.0) + 127.5  # Reverse the normalization
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
        
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

        for attack_type in ["FGSM"]:
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
    framework = SphereAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_path='E:/AdversarialAttack-2/Model/Weights/sphere20a_20171020.pth'
    )
    
    results = framework.run_evaluation()
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    '''
    if len(framework.pairs) > 0:
        img1_path, img2_path, label = framework.pairs[0]  # Get the first pair
        print(f"Using image pair: {img1_path}, {img2_path}, Same person? {label==1}")
    else:
        print("No image pairs found. Please check your dataset path.")
        sys.exit(1)
    
    # Generate adversarial examples using different attack methods
    attacks = {
        "FGSM": framework.generateFGSMAttack,
        "PGD": framework.generatePGDAttack,
        "BIM": framework.generateBIMAttack,
        "MIFGSM": framework.generateMIFGSMAttack,
        "Square": framework.generateSquareAttack,
        "SPSA": framework.generateSPSAAttack,
        "CW": framework.generateCWAttack
    }
    
    # Select one attack to debug
    attack_name = "Square"  # Change this to debug different attacks
    attack_func = attacks[attack_name]
    
    # Generate the adversarial example
    print(f"Generating {attack_name} adversarial example...")
    adv_img_path = attack_func(img1_path, img2_path, label)
    print(f"Adversarial example saved to: {adv_img_path}")
    
    # Load original images and adversarial image for display
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    adv_img = cv2.imread(adv_img_path)
    
    # Convert from BGR to RGB for display
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    adv_img = cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB)
    
    # Resize for uniform display
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    adv_img = cv2.resize(adv_img, (224, 224))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display the images
    axes[0].imshow(img1)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(adv_img)
    axes[1].set_title(f"{attack_name} Adversarial")
    axes[1].axis('off')
    
    axes[2].imshow(img2)
    axes[2].set_title("Target Image")
    axes[2].axis('off')
    
    # Calculate and display the verification results
    # Looking at the original function, verify_pair returns the comparison with a threshold
    # So the results are already boolean (True for match, False for no match)
    orig_match = framework.verify_pair(img1_path, img2_path)
    adv_match = framework.verify_pair(adv_img_path, img2_path)
    
    # Define attack success based on the goal
    # If label=1 (same person): attack success means changing match to no match
    # If label=0 (different people): attack success means changing no match to match
    if label == 1:
        attack_success = orig_match and not adv_match  # Changed from match to no match
    else:
        attack_success = not orig_match and adv_match  # Changed from no match to match
    
    # Calculate perturbation magnitude
    perturbation = adv_img.astype(np.float32) - img1.astype(np.float32)
    l2_norm = np.sqrt(np.sum(perturbation**2))
    linf_norm = np.max(np.abs(perturbation))
    
    # Display results as text
    result_text = (
        f"Attack: {attack_name}\n"
        f"Original pair verification: {'Match' if orig_match else 'No Match'}\n"
        f"Adversarial pair verification: {'Match' if adv_match else 'No Match'}\n"
        f"Attack {'successful' if attack_success else 'failed'}\n"
        f"L2 perturbation: {l2_norm:.2f}\n"
        f"L∞ perturbation: {linf_norm:.2f}"
    )
    
    plt.figtext(0.5, 0.01, result_text, ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the text
    plt.show()
    
    # Also display the perturbation itself (magnified for visibility)
    plt.figure(figsize=(10, 5))
    
    # Normalize perturbation for better visualization
    perturbation_vis = np.abs(perturbation)
    perturbation_vis = perturbation_vis / np.max(perturbation_vis) * 255
    
    plt.imshow(perturbation_vis.astype(np.uint8))
    plt.title(f"Perturbation Map ({attack_name})")
    plt.colorbar(label="Magnitude")
    plt.show()
    
    print(f"Attack was {'successful' if attack_success else 'unsuccessful'}")
    os.remove(adv_img_path)
    '''