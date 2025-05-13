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
        
        
        adaface_models = {'ir_50':"./Model/Weights/adaface_ir50_ms1mv2.ckpt",}
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
    def to_input(self,pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.from_numpy(np.array([brg_img.transpose(2,0,1)])).float().to(self.device)
        return tensor
    def verify_pair(self, img1_path, img2_path, threshold=0.3):
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]  
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]    
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        is_adv_example = any(suffix in file1 for suffix in ['_fgsm_adv.jpg', '_pgd_adv.jpg', '_bim_adv.jpg', 
                                                        '_mifgsm_adv.jpg', '_cw_adv.jpg', '_spsa_adv.jpg', 
                                                        '_square_adv.jpg'])
        
        
        if is_adv_example:
            base_name = '_'.join(file1.split('_')[:-2]) 
            original_file = base_name + '.jpg'
            landmark_key1 = f"{person1}/{original_file}"
        else:
            landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
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
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        
        img2_tensor = self.to_input(img2)
        
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device).requires_grad_(True)
        
        
        with torch.no_grad():
            
            feature2, _ = self.model(img2_tensor)
        
        
        feature1, _ = self.model(img1_tensor)
                
        
        if label == 1:  
            loss = -torch.mm(feature1, feature2.T)
        else:  
            loss = torch.mm(feature1, feature2.T)
        
        grad = torch.autograd.grad(loss, img1_tensor)[0]
        
        
        perturbed_image = img1_tensor.detach() + epsilon * torch.sign(grad)
        
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
        
        adv_img = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        adv_img = (adv_img + 0.5) * 255.0
        adv_img = np.clip(adv_img[:,:,::-1], 0, 255).astype(np.uint8)  
        
        output_path = img1_path.replace('.jpg', '_fgsm_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generatePGDAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, steps=20):
        if alpha is None:
            alpha = epsilon/10
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        img2_tensor = self.to_input(img2)
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        with torch.no_grad():
            
            feature2, _ = self.model(img2_tensor)
        
        
        perturbed_image = img1_tensor.clone().detach()
        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon/2, epsilon/2)
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0).detach()
        
        
        for _ in range(steps):
            perturbed_image.requires_grad = True
            
            
            feature_adv, _ = self.model(perturbed_image)
            
            
            similarity = torch.mm(feature_adv, feature2.T)
            

            if label == 1: 
                loss = -similarity
            else:  
                loss = similarity
            
            
            grad = torch.autograd.grad(loss, perturbed_image)[0]
            
            
            perturbed_image = perturbed_image.detach() + alpha * torch.sign(grad)
            
            
            delta = torch.clamp(perturbed_image - img1_tensor, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(img1_tensor + delta, min=-1.0, max=1.0).detach()
        
        
        adv_img = perturbed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        adv_img = (adv_img + 0.5) * 255.0
        adv_img = np.clip(adv_img[:,:,::-1], 0, 255).astype(np.uint8)  
        
        
        output_path = img1_path.replace('.jpg', '_pgd_adv.jpg')
        cv2.imwrite(output_path, adv_img)
        
        return output_path
    def generateBIMAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, iterations=20):
        if alpha is None:
            alpha = epsilon/10
            
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        img2_tensor = self.to_input(img2)
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        
        adv_img = img1_tensor.clone().detach()
        ori_img = img1_tensor.clone().detach()
        
        
        for _ in range(iterations):
            
            adv_img.requires_grad = True
            
            
            feature_adv, _ = self.model(adv_img)
            
            similarity = torch.mm(feature_adv, feature2.T)
            
            
            if label == 1:  
                loss = -similarity
            else:  
                loss = similarity
            
            
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            adv_img = adv_img.detach()
            
            adv_img = adv_img + alpha * torch.sign(grad)
            

            a = torch.clamp(ori_img - epsilon, min=-1.0)
            b = (adv_img >= a).float() * adv_img + (adv_img < a).float() * a
            c = (b > ori_img + epsilon).float() * (ori_img + epsilon) + (b <= ori_img + epsilon).float() * b
            adv_img = torch.clamp(c, min=-1.0, max=1.0).detach()
        
        
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  
        
        output_path = img1_path.replace('.jpg', '_bim_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateMIFGSMAttack(self, img1_path, img2_path, label=None, epsilon=8/255, alpha=None, iterations=20, decay_factor=0.9):
        if alpha is None:
            alpha = epsilon/10
            
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        
        img2_tensor = self.to_input(img2)
        
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        
        adv_img = img1_tensor.clone().detach()
        
        
        momentum = torch.zeros_like(img1_tensor).to(self.device)
        
        
        for _ in range(iterations):
            
            adv_img.requires_grad = True
            
            
            feature_adv, _ = self.model(adv_img)
            
            
            similarity = torch.mm(feature_adv, feature2.T)
            
            if label == 1:  
                loss = -similarity
            else:  
                loss = similarity
            
            
            grad = torch.autograd.grad(loss, adv_img)[0]
            
            
            adv_img = adv_img.detach()
            
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / (grad_norm + 1e-10)  
            
            
            momentum = decay_factor * momentum + grad
            
            
            adv_img = adv_img + alpha * momentum.sign()
            
            
            delta = torch.clamp(adv_img - img1_tensor, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(img1_tensor + delta, min=-1.0, max=1.0).detach()
        
        
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  
        
        
        output_path = img1_path.replace('.jpg', '_mifgsm_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateCWAttack(self, img1_path, img2_path, label=None, c=1.0, kappa=0, steps=30, lr=0.01, threshold=0.3):
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        
        img2_tensor = self.to_input(img2)
        
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        
        def tanh_space(x):
            return 0.5 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x):
            
            x_clamped = torch.clamp(x, min=-0.999, max=0.999)
            return torch.atanh(x_clamped)
        
        
        
        img1_norm = (img1_tensor + 1.0) / 2.0
        w = inverse_tanh_space(2 * img1_norm - 1).detach()
        w.requires_grad = True
        
        
        optimizer = torch.optim.Adam([w], lr=lr)
        
        
        best_adv_images = img1_tensor.clone().detach()
        best_L2 = 1e10 * torch.ones(len(img1_tensor)).to(self.device)
        prev_cost = 1e10
        
        
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        
        
        for step in range(steps):
            
            adv_images_norm = tanh_space(w)
            adv_images = adv_images_norm * 2.0 - 1.0  
            
            
            current_L2 = MSELoss(Flatten(adv_images_norm), Flatten(img1_norm)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            
            feature_adv, _ = self.model(adv_images)
            
            
            similarity = torch.mm(feature_adv, feature2.T)
            
            
            if label == 1:  
                f_loss = torch.clamp(similarity - threshold + kappa, min=0)
            else:  
                f_loss = torch.clamp(threshold - similarity + kappa, min=0)
            
            
            cost = L2_loss + c * f_loss
            
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            
            if label == 1:  
                condition = (similarity < threshold).float()
            else:  
                condition = (similarity > threshold).float()
            
            
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            
            
            mask = mask.view([-1] + [1] * (len(adv_images.shape) - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            
            
            if step % max(steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    break
                prev_cost = cost.item()
        
        
        perturbed_image = best_adv_images[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        perturbed_image = (perturbed_image + 0.5) * 255.0
        
        
        perturbed_image = np.nan_to_num(perturbed_image, nan=0.0, posinf=255.0, neginf=0.0)
        
        
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)
        
        
        output_path = img1_path.replace('.jpg', '_cw_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSPSAAttack(self, img1_path, img2_path, label=None, epsilon=8/255, iterations=100, learning_rate=0.01, delta=0.01):
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        
        img2_tensor = self.to_input(img2)
        
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        
        adv_img = img1_tensor.clone().detach()
        
        
        def compute_loss(perturbed_img):
            with torch.no_grad():
                feature_perturbed, _ = self.model(perturbed_img)
                
                similarity = torch.mm(feature_perturbed, feature2.T)
                
                if label == 1:  
                    return similarity  
                else:  
                    return -similarity  
        
        
        for i in range(iterations):
            
            bernoulli = torch.randint(0, 2, adv_img.shape).to(self.device) * 2 - 1  
            
            
            pos_perturbed = adv_img + delta * bernoulli
            pos_perturbed = torch.clamp(pos_perturbed, -1.0, 1.0)
            loss_pos = compute_loss(pos_perturbed)
            
            neg_perturbed = adv_img - delta * bernoulli
            neg_perturbed = torch.clamp(neg_perturbed, -1.0, 1.0)
            loss_neg = compute_loss(neg_perturbed)
            
            
            gradient_estimate = (loss_pos - loss_neg) / (2 * delta)
            
            
            
            adv_img = adv_img - learning_rate * gradient_estimate * bernoulli
            
            
            delta_img = torch.clamp(adv_img - img1_tensor, min=-epsilon, max=epsilon)
            adv_img = img1_tensor + delta_img
            adv_img = torch.clamp(adv_img, -1.0, 1.0)
            
            
            learning_rate = learning_rate * 0.99
        
        
        perturbed_image = adv_img[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  
        
        
        output_path = img1_path.replace('.jpg', '_spsa_adv.jpg')
        cv2.imwrite(output_path, perturbed_image)
        
        return output_path
    def generateSquareAttack(self, img1_path, img2_path, label=None, n_iters=1000, p_init=0.1, epsilon=8/255, threshold=0.3):
        
        img1_parts = img1_path.split(os.sep)
        img2_parts = img2_path.split(os.sep)
        
        person1 = img1_parts[-2]
        person2 = img2_parts[-2]
        
        file1 = img1_parts[-1]
        file2 = img2_parts[-1]
        
        
        landmark_key1 = f"{person1}/{file1}"
        landmark_key2 = f"{person2}/{file2}"
        
        
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        img1 = self.alignment(img1_orig, self.landmark[landmark_key1])
        img2 = self.alignment(img2_orig, self.landmark[landmark_key2])
        
        
        img2_tensor = self.to_input(img2)
        
        
        np_img1 = np.array(img1)
        brg_img1 = ((np_img1[:,:,::-1] / 255.) - 0.5) / 0.5
        img1_tensor = torch.from_numpy(np.array([brg_img1.transpose(2,0,1)])).float().to(self.device)
        
        
        with torch.no_grad():
            feature2, _ = self.model(img2_tensor)
        
        
        x_adv = img1_tensor.clone().detach()
        
        
        def compute_similarity(x):
            with torch.no_grad():
                feature_x, _ = self.model(x)
                similarity = torch.mm(feature_x, feature2.T)
                return similarity.item()
        
        
        if label == 1:  
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim < curr_sim
        else:  
            best_similarity = compute_similarity(x_adv)
            is_better = lambda new_sim, curr_sim: new_sim > curr_sim
        
        
        _, _, h, w = img1_tensor.shape  
        
        
        for i in range(n_iters):
            
            p = p_init * (1 - i / n_iters)**0.5
            
            s = int(round(np.sqrt(p * h * w)))
            s = max(1, min(s, h, w))
            
            h_start = np.random.randint(0, h - s + 1)
            w_start = np.random.randint(0, w - s + 1)
            
            channel = np.random.choice([-1, 0, 1, 2])  
            
            
            x_new = x_adv.clone().detach()
            
            
            if channel == -1:  
                noise = torch.empty((1, 3, s, s), device=self.device).uniform_(-2*epsilon, 2*epsilon)
                x_new[0, :, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, :, h_start:h_start+s, w_start:w_start+s] + noise[0], -1.0, 1.0)
            else:  
                noise = torch.empty((1, s, s), device=self.device).uniform_(-2*epsilon, 2*epsilon)
                x_new[0, channel, h_start:h_start+s, w_start:w_start+s] = torch.clamp(
                    img1_tensor[0, channel, h_start:h_start+s, w_start:w_start+s] + noise[0], -1.0, 1.0)
            
            
            new_similarity = compute_similarity(x_new)
            
            
            if is_better(new_similarity, best_similarity):
                x_adv = x_new
                best_similarity = new_similarity
            
            
            if (label == 1 and best_similarity < threshold) or (label == 0 and best_similarity > threshold):
                break
        
        
        perturbed_image = x_adv[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        perturbed_image = (perturbed_image + 0.5) * 255.0
        perturbed_image = np.clip(perturbed_image[:,:,::-1], 0, 255).astype(np.uint8)  
        
        
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
                
                
                if os.path.exists(adv_img_path):
                    os.remove(adv_img_path)                
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
    def save_l2_to_txt(self, filename="Verification_metric/SimScore_values_Ada.txt"):
        with open(filename, 'w') as f:
            for value in self.SimScore:
                f.write(f"{value}\n")
        print(f"Saved {len(self.SimScore)} Similarity Score values to {filename}")


if __name__ == "__main__":
    framework = AdaFaceAttackFramework(
        data_dir='./lfw_funneled',
    )
    results = framework.run_evaluation()
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
