import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import os,sys
import torch.nn as nn
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Model.Architecture.OpenFaceArchitecture import netOpenFace
import matplotlib.pyplot as plt

class OpenFaceAttackFramework:
    def __init__(self, data_dir, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        # Initialize model
        self.model = netOpenFace(useCuda = True, gpuDevice = 0)
        self.model.load_state_dict(torch.load("E:/AdversarialAttack-2/Model/Weights/openface.pth"))
        self.model.eval()
    def ReadImage(self, pathname):
        img = cv2.imread(pathname)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        I_ = torch.from_numpy(img).unsqueeze(0)
        I_ = I_.cuda()
        return I_       
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
    def cosine_similarity(self, feat1, feat2):
        # Compute cosine similarity between two feature vectors
        return torch.nn.functional.cosine_similarity(feat1, feat2, dim=0)
    def verify_pair(self, img1_path, img2_path, threshold=0.7):
        img1 = self.ReadImage(pathname=img1_path)
        img2 = self.ReadImage(pathname=img2_path)
        
        # Combine images into a batch
        I_ = torch.cat([img1, img2], 0)
        
        # Get embeddings from the model
        with torch.no_grad():
            _, features = self.model(I_)
        
        # Compute cosine similarity
        similarity = self.cosine_similarity(features[0], features[1])
        
        distance = 1.0 - similarity.item()
        is_same_person = distance < threshold

        print(distance)
        return is_same_person
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
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()
        
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
            perturbed_image = perturbed_image.detach() + alpha * grad.sign()  # Note the minus sign
            
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
            adv_img = adv_img + alpha * torch.sign(grad)
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
            
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            # Detach from computation graph
            adv_img = adv_img.detach()
            
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / grad_norm 
            
            # Update momentum term
            grad = grad + momentum * decay_factor
            momentum = grad            
            adv_img = adv_img + alpha * grad.sign()
            
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
            return torch.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))
        
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
        
        output_path = img1_path.replace('.jpg', '_cw_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        
        return output_path
    def generateSPSAAttack(self, img1_path, img2_path, label=None):
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

        # SPSA parameters
        epsilon = 8/255        # Total perturbation constraint
        iterations = 100       # Number of attack iterations
        learning_rate = 0.01   # Learning rate for optimization
        delta = 0.01           # Perturbation size for gradient estimation
        
        # Extract features from target image
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Initialize adversarial example with the original image
        adv_img = img1.clone().detach()
        
        # Function to compute loss based on goal
        def compute_loss(perturbed_img):
            with torch.no_grad():
                feat = self.model.get_features(perturbed_img)
                feat = F.normalize(feat, p=2, dim=1)
                distance = torch.norm(feat - feat2, p=2)
                
                # Define loss based on attack goal
                if label == 1:  # Decrease distance (make different people look same)
                    return distance
                else:  # Increase distance (make same person look different)
                    return -distance
        
        # SPSA attack loop
        for i in range(iterations):
            # Create random perturbation direction (Bernoulli distribution {-1, 1})
            bernoulli = torch.randint(0, 2, adv_img.shape).to(self.device) * 2 - 1  # -1 or 1
            
            # Evaluate loss at points in both positive and negative directions
            pos_perturbed = adv_img + delta * bernoulli
            pos_perturbed = torch.clamp(pos_perturbed, 0, 255)
            loss_pos = compute_loss(pos_perturbed)
            
            neg_perturbed = adv_img - delta * bernoulli
            neg_perturbed = torch.clamp(neg_perturbed, 0, 255)
            loss_neg = compute_loss(neg_perturbed)
            
            # Estimate gradient using finite differences
            gradient_estimate = (loss_pos - loss_neg) / (2 * delta)
            
            # Update the adversarial example in the direction of the estimated gradient
            # Note the sign: We use minus for gradient descent direction
            if label == 1:  # Minimize distance
                update_direction = -1
            else:  # Maximize distance
                update_direction = 1
                
            # Apply estimated gradient to update the adversarial example
            adv_img = adv_img + update_direction * learning_rate * gradient_estimate * bernoulli
            
            # Project back to epsilon-ball around original image and ensure valid pixel range
            delta_img = torch.clamp(adv_img - img1, min=-epsilon, max=epsilon)
            adv_img = img1 + delta_img
            adv_img = torch.clamp(adv_img, 0, 255)
            
            # Optionally reduce learning rate over time (learning rate decay)
            learning_rate = learning_rate * 0.99
        
        # Convert to image and save
        adv_output = adv_img[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_output += np.array([129.1863, 104.7624, 93.5940])
        adv_output = np.clip(adv_output, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_spsa_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        return output_path
    def generateSquareAttack(self, img1_path, img2_path, label=None, n_iters=1000, p_init=0.1):
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

        # Square attack parameters
        epsilon = 8/255
        
        # Extract features from target image
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Initialize adversarial example with the original image
        x_adv = img1.clone().detach()
        
        # Function to evaluate distance
        def compute_distance(x):
            with torch.no_grad():
                feat = self.model.get_features(x)
                feat = F.normalize(feat, p=2, dim=1)
                distance = torch.norm(feat - feat2, p=2)
                return distance.item()
        
        # Determine the best distance based on the attack goal
        if label == 1:  # Make different people look same (minimize distance)
            best_distance = compute_distance(x_adv)
            is_better = lambda new_dist, curr_dist: new_dist < curr_dist
        else:  # Make same person look different (maximize distance)
            best_distance = -compute_distance(x_adv)
            is_better = lambda new_dist, curr_dist: new_dist > curr_dist
        
        # Reshape image for easier manipulation
        h, w = 224, 224  # Image height and width
        
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
            
            # Generate perturbation
            noise = torch.empty((1, 1 if channel != -1 else 3, s, s), device=self.device).uniform_(-epsilon, epsilon)
            
            # Apply the perturbation to the selected region
            if channel == -1:  # Apply to all channels
                x_new[0, :, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1[0, :, h_start:h_start+s, w_start:w_start+s] + noise, 0, 255)
            else:  # Apply to specific channel
                x_new[0, channel, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1[0, channel, h_start:h_start+s, w_start:w_start+s] + noise.squeeze(1), 0, 255)
            
            # Calculate new distance
            if label == 1:  # Minimize distance
                new_distance = compute_distance(x_new)
            else:  # Maximize distance
                new_distance = -compute_distance(x_new)
            
            # Update best adversarial example if the perturbation improves it
            if is_better(new_distance, best_distance):
                x_adv = x_new
                best_distance = new_distance
                
            # Early stopping if we've reached a very good solution
            if (label == 1 and best_distance < 0.5) or (label == 0 and -best_distance > 2.0):
                break
        # Convert to image and save
        adv_output = x_adv[0].permute(1, 2, 0).contiguous().cpu().numpy()
        adv_output += np.array([129.1863, 104.7624, 93.5940])
        adv_output = np.clip(adv_output, 0, 255).astype(np.uint8)
        
        output_path = img1_path.replace('.jpg', '_square_adv.jpg')
        cv2.imwrite(output_path, adv_output)
        
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
        for attack_type in []:
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
    framework = OpenFaceAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
    )
    results = framework.run_evaluation()
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


    """
    if len(framework.pairs) > 0:
        img1_path, img2_path, label = framework.pairs[50]  # Get the first pair
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
    attack_name = "SPSA"  # Change this to debug different attacks
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
    """