import os
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, R2U_NetPP, AttU_Net, R2AttU_Net
import csv


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob
        self.model_type = config.model_type
        self.t = config.t

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path

        self.mode = config.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'R2U_Net++':
            self.unet = R2U_NetPP(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)

        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        print(model)
        print(name)
        print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters())))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def save_samples(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.unet.eval()

        with torch.no_grad():
            for images, GT in self.test_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = torch.sigmoid(self.unet(images))
                break

        n = min(3, images.size(0))
        for i in range(n):
            img = images[i].cpu()
            gt = GT[i].cpu()
            sr = SR[i].cpu()

            # make GT & SR 3-channel for visualization
            if gt.size(0) == 1:
                gt_viz = gt.repeat(3,1,1)
            else:
                gt_viz = gt
            if sr.size(0) == 1:
                sr_viz = sr.repeat(3,1,1)
            else:
                sr_viz = sr

            img = (img.permute(1,2,0) * 255).byte().numpy()
            gt_viz = (gt_viz.permute(1,2,0) * 255).byte().numpy()
            sr_viz = (sr_viz.permute(1,2,0) * 255).byte().numpy()

            combined = np.vstack([img, gt_viz, sr_viz])
            Image.fromarray(combined).save(os.path.join(save_dir, f"sample_{i+1}.png"))

    def train(self):
        os.makedirs(self.model_path, exist_ok=True)
        unet_path = os.path.join(self.model_path, f'{self.model_type}-{self.num_epochs}-{self.lr:.4f}-{self.augmentation_prob:.4f}.pkl')

        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
            print(f'{self.model_type} loaded from {unet_path}')
            return

        best_unet_score = -1.0

        for epoch in range(self.num_epochs):
            # Training
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            self.unet.train()
            epoch_loss = 0.0

            acc = SE = SP = PC = F1 = JS = DC = 0.0
            length = 0

            for images, GT in self.train_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)

                # GT is binary 0/1; if empty mask then skip training step
                if GT.sum().item() == 0:
                    continue

                SR = torch.sigmoid(self.unet(images))
                SR_flat = SR.view(SR.size(0), -1)
                GT_flat = GT.view(GT.size(0), -1)

                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

            if length > 0:
                acc /= length; SE /= length; SP /= length; PC /= length
                F1 /= length; JS /= length; DC /= length
            else:
                acc = SE = SP = PC = F1 = JS = DC = None

            print(f'Loss: {epoch_loss:.4f}\n[Training] Acc: {acc}, SE: {SE}, SP: {SP}, PC: {PC}, F1: {F1}, JS: {JS}, DC: {DC}')

            # Validation
            self.unet.eval()
            acc = SE = SP = PC = F1 = JS = DC = 0.0
            length = 0

            with torch.no_grad():
                for images, GT in self.valid_loader:
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    if GT.sum().item() == 0:
                        continue

                    SR = torch.sigmoid(self.unet(images))
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length += images.size(0)

            if length > 0:
                acc /= length; SE /= length; SP /= length; PC /= length
                F1 /= length; JS /= length; DC /= length
            else:
                acc = SE = SP = PC = F1 = JS = DC = None

            print(f'[Validation] Acc: {acc}, SE: {SE}, SP: {SP}, PC: {PC}, F1: {F1}, JS: {JS}, DC: {DC}')

            # Decay learning rate
            self.scheduler.step()
            print("Current LR:", self.optimizer.param_groups[0]["lr"])

            unet_score = (JS if JS is not None else 0.0) + (DC if DC is not None else 0.0)
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                torch.save(self.unet.state_dict(), unet_path)
                print(f"Saved best model (score={best_unet_score:.4f}) to {unet_path}")

        # Test (load best model)
        self.unet.load_state_dict(torch.load(unet_path))
        self.unet.eval()

        acc = SE = SP = PC = F1 = JS = DC = 0.0
        length = 0

        with torch.no_grad():
            for images, GT in self.test_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)

                if GT.sum().item() == 0:
                    continue

                SR = torch.sigmoid(self.unet(images))
                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

        if length > 0:
            acc /= length; SE /= length; SP /= length; PC /= length
            F1 /= length; JS /= length; DC /= length
        else:
            acc = SE = SP = PC = F1 = JS = DC = None

        # save sample outputs
        save_dir = os.path.join(self.model_path, f"{self.model_type}-{self.num_epochs}-{self.lr:.6f}")
        self.save_samples(save_dir)
        os.makedirs(self.result_path, exist_ok=True)
        with open(os.path.join(self.result_path, 'result.csv'), 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr])
