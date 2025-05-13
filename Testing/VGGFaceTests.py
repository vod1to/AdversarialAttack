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
import matplotlib.pyplot as plt

class VGGAttackFramework:
    def __init__(self, data_dir, model_path, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        
        self.model = VGGFace().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.L2 = []        
    def prepare_pairs(self):
        pairs = []
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        
        for person in classes:
            person_dir = os.path.join(self.data_dir, person)
            images = os.listdir(person_dir)
            if len(images) >= 2:
                img1 = os.path.join(person_dir, images[0])
                img2 = os.path.join(person_dir, images[1])
                pairs.append((img1, img2, 1))
            if len(pairs) == 1:
                break
        
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
            if len(pairs) == 2:
                break

        return pairs
    def verify_pair(self, img1_path, img2_path, threshold=1.2):
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
        
        self.model.eval()      
        with torch.no_grad():
            
            feat1 = self.model.get_features(img1)
            feat2 = self.model.get_features(img2)
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            
            l2_distance = torch.norm(feat1 - feat2, p=2).item()
        self.L2.append(l2_distance)
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
        epsilon = 8/255  
    
        self.model.eval()
        feat1 = self.model.get_features(img1_adv)
        feat2 = self.model.get_features(img2)
    
        
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        distance = torch.norm(feat1 - feat2, p=2)
        
        if label == 1:  
            loss = distance
        else:  
            loss = -distance
        

        grad_sign = torch.autograd.grad(loss, img1_adv)[0]
        
        perturbed_image = img1 + epsilon * grad_sign.sign()
        
        perturbed_image = torch.clamp(perturbed_image, 0, 255)
            
        
        
        adv_img = perturbed_image[0].permute(1, 2, 0).cpu().numpy()
        adv_img += np.array([129.1863, 104.7624, 93.5940]) 
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        
        
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

        epsilon = 8/255  
        alpha = epsilon/10  
        steps = 20  

        self.model.eval()
        perturbed_image = img1.clone().detach()
        
        
        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()
        
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)


        
        for _ in range(steps):
            
            perturbed_image.requires_grad = True
            
            
            feat1 = self.model.get_features(perturbed_image)
            
            
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            
            distance = torch.norm(feat1 - feat2, p=2)
            
            if label == 1:  
                loss = distance
            else:  
                loss = -distance
            
            
            grad = torch.autograd.grad(loss, perturbed_image)[0]
        
            
            perturbed_image = perturbed_image.detach() + alpha * grad.sign() 
            
            
            delta = torch.clamp(perturbed_image - img1, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(img1 + delta, min=0, max=255).detach()
        
        
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

        
        epsilon = 8/255      
        alpha = epsilon/10   
        iterations = 20      

        
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        adv_img = img1.clone().detach()
        ori_img = img1.clone().detach()
        
        
        for i in range(iterations):
            
            adv_img.requires_grad = True
            
            
            feat1 = self.model.get_features(adv_img)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            
            distance = torch.norm(feat1 - feat2, p=2)
            
            if label == 1:  
                loss = distance
            else:  
                loss = -distance
            
            
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            
            adv_img = adv_img.detach()
            
            adv_img = adv_img + alpha * torch.sign(grad)
            a = torch.clamp(ori_img - epsilon, min=0)
            b = (adv_img >= a).float() * adv_img + (adv_img < a).float() * a
            c = (b > ori_img + epsilon).float() * (ori_img + epsilon) + (b <= ori_img + epsilon).float() * b
            
            
            adv_img = torch.clamp(c, min=0, max=255).detach()
        
        
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

        
        epsilon = 8/255       
        alpha = epsilon/10    
        iterations = 20       
        decay_factor = 0.9    

        
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        adv_img = img1.clone().detach()
        
        
        momentum = torch.zeros_like(img1).to(self.device)
        
        
        for i in range(iterations):
            
            adv_img.requires_grad = True
            
            
            feat1 = self.model.get_features(adv_img)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            
            distance = torch.norm(feat1 - feat2, p=2)
            
            
            if label == 1:  
                loss = distance
            else:  
                loss = -distance
            
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            
            adv_img = adv_img.detach()
            
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / grad_norm 
            
            
            grad = grad + momentum * decay_factor
            momentum = grad            
            adv_img = adv_img + alpha * grad.sign()
            
            delta = torch.clamp(adv_img - img1, min=-epsilon, max=epsilon)
            adv_img = img1 + delta
            adv_img = torch.clamp(adv_img, min=0, max=255)
        
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
        
        
        self.model.eval()
        
        
        def tanh_space(x):
            return 0.5 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x):
            return torch.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))
        
        
        w = inverse_tanh_space(img1 / 255.0).detach()  
        w.requires_grad = True
        
        
        optimizer = torch.optim.Adam([w], lr=lr)
        
        
        best_adv_images = img1.clone().detach()
        best_L2 = 1e10 * torch.ones((len(img1))).to(self.device)
        prev_cost = 1e10
        
        
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        
        
        for step in range(steps):
            
            adv_images_norm = tanh_space(w)
            adv_images = adv_images_norm * 255.0  
            
            
            current_L2 = MSELoss(Flatten(adv_images_norm), Flatten(img1 / 255.0)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            
            feat1 = self.model.get_features(adv_images)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
            
            distance = torch.norm(feat1 - feat2, p=2, dim=1)
            threshold = 1.2

            
            if label == 1: 
                f_loss = torch.clamp(distance - kappa, min=0).sum()
            else:  
                f_loss = torch.clamp(threshold - distance + kappa, min=0).sum()
            
            
            cost = L2_loss + c * f_loss
            
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            

            if label == 1:  
                condition = (distance < threshold).float()
            else: 
                condition = (distance > threshold).float()
            
            
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            
            
            mask = mask.view([-1] + [1] * (len(adv_images.shape) - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            
            
            if step % max(steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    break
                prev_cost = cost.item()
        
        
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

        
        epsilon = 8/255        
        iterations = 100       
        learning_rate = 0.01   
        delta = 0.01           
        
        
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        adv_img = img1.clone().detach()
        
        
        def compute_loss(perturbed_img):
            with torch.no_grad():
                feat = self.model.get_features(perturbed_img)
                feat = F.normalize(feat, p=2, dim=1)
                distance = torch.norm(feat - feat2, p=2)
                
                if label == 1:  
                    return distance
                else:  
                    return -distance
        
        
        for i in range(iterations):
            
            bernoulli = torch.randint(0, 2, adv_img.shape).to(self.device) * 2 - 1  
            
            
            pos_perturbed = adv_img + delta * bernoulli
            pos_perturbed = torch.clamp(pos_perturbed, 0, 255)
            loss_pos = compute_loss(pos_perturbed)
            
            neg_perturbed = adv_img - delta * bernoulli
            neg_perturbed = torch.clamp(neg_perturbed, 0, 255)
            loss_neg = compute_loss(neg_perturbed)
            
            
            gradient_estimate = (loss_pos - loss_neg) / (2 * delta)
            
            
            
            if label == 1: 
                update_direction = -1
            else:  
                update_direction = 1
                
            
            adv_img = adv_img + update_direction * learning_rate * gradient_estimate * bernoulli
            
            
            delta_img = torch.clamp(adv_img - img1, min=-epsilon, max=epsilon)
            adv_img = img1 + delta_img
            adv_img = torch.clamp(adv_img, 0, 255)
            
            
            learning_rate = learning_rate * 0.99
        
        
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

        
        epsilon = 8/255
        
        
        self.model.eval()
        with torch.no_grad():
            feat2 = self.model.get_features(img2)
            feat2 = F.normalize(feat2, p=2, dim=1)
        
        
        x_adv = img1.clone().detach()
        
        
        def compute_distance(x):
            with torch.no_grad():
                feat = self.model.get_features(x)
                feat = F.normalize(feat, p=2, dim=1)
                distance = torch.norm(feat - feat2, p=2)
                return distance.item()
        
        
        if label == 1:  
            best_distance = compute_distance(x_adv)
            is_better = lambda new_dist, curr_dist: new_dist < curr_dist
        else:  
            best_distance = -compute_distance(x_adv)
            is_better = lambda new_dist, curr_dist: new_dist > curr_dist
        
        
        h, w = 224, 224  
        
        
        for i in range(n_iters):
            
            p = p_init * (1 - i / n_iters)**0.5
            
            
            s = int(round(np.sqrt(p * h * w)))
            s = max(1, min(s, h, w))
            
            
            h_start = np.random.randint(0, h - s + 1)
            w_start = np.random.randint(0, w - s + 1)
            
            
            channel = np.random.choice([-1, 0, 1, 2])  
            
            
            x_new = x_adv.clone().detach()
            
            
            noise = torch.empty((1, 1 if channel != -1 else 3, s, s), device=self.device).uniform_(-epsilon, epsilon)
            
            
            if channel == -1:  
                x_new[0, :, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1[0, :, h_start:h_start+s, w_start:w_start+s] + noise, 0, 255)
            else:  
                x_new[0, channel, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1[0, channel, h_start:h_start+s, w_start:w_start+s] + noise.squeeze(1), 0, 255)
            
            
            if label == 1:  
                new_distance = compute_distance(x_new)
            else:  
                new_distance = -compute_distance(x_new)
            
            
            if is_better(new_distance, best_distance):
                x_adv = x_new
                best_distance = new_distance
                
            
            if (label == 1 and best_distance < 0.5) or (label == 0 and -best_distance > 2.0):
                break
        
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
                
                prediction = self.verify_pair(adv_img_path, img2_path)
                
                
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
                
                
                
                
                
            except Exception as e:
                print(f"Error processing pair: {e}")
        
        
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
        

        for attack_type in ["FGSM", "PGD", "BIM", "MIFGSM", "CW","SPSA","Square"]:
            print(f"\nEvaluating {attack_type} attack...")
            results[attack_type] = self.evaluate_attack(attack_type)
       
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
    def save_l2_to_txt(self, filename="Verification_metric/l2_values_VGG.txt"):
        with open(filename, 'w') as f:
            for value in self.L2:
                f.write(f"{value}\n")
        print(f"Saved {len(self.L2)} L2 values to {filename}")

if __name__ == "__main__":
    framework = VGGAttackFramework(
        data_dir='./lfw_funneled',
        model_path='./Model/Weights/vgg_face_dag.pth'
    )
    results = framework.run_evaluation()
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


