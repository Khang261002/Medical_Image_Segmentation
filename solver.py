import os
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from evaluation import *
from network import *
import csv


class DiceLoss(nn.Module):
    def forward(self, logits, targets, eps=1e-7):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum()
        den = probs.sum() + targets.sum() + eps
        return 1 - num / den


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.dataset = config.dataset

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()
        self.dice = DiceLoss()
        self.criterion = lambda logits, targets: (
            F.binary_cross_entropy_with_logits(logits, targets) +
            self.dice(logits, targets)
        )
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

        init_weights(self.unet, init_type='kaiming')
        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(100, self.num_epochs), eta_min=1e-6)
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        print(model)
        print(name)
        print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters())))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.optimizer.zero_grad()

    def save_samples(self, save_dir, num_samples=3):
        os.makedirs(save_dir, exist_ok=True)
        self.unet.eval()

        collected_images = []
        collected_gts = []
        collected_srs = []

        # ---- Collect exactly (num_samples) samples ----
        with torch.no_grad():
            for images, GT in self.test_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = torch.sigmoid(self.unet(images))

                for i in range(images.size(0)):
                    collected_images.append(images[i].cpu())
                    collected_gts.append(GT[i].cpu())
                    collected_srs.append(SR[i].cpu())

                    if len(collected_images) >= num_samples:
                        break

                if len(collected_images) >= num_samples:
                    break

        # ---- Save all samples with max number of (num_samples) samples ----
        for idx, (img, gt, sr) in enumerate(zip(collected_images, collected_gts, collected_srs)):
            # make GT & SR 3-channel
            if gt.size(0) == 1:
                gt_viz = gt.repeat(3,1,1)
            else:
                gt_viz = gt
            if sr.size(0) == 1:
                sr_viz = sr.repeat(3,1,1)
            else:
                sr_viz = sr

            img = (img.permute(1,2,0) * 255).clamp(0,255).byte().numpy()
            gt_viz = (gt_viz.permute(1,2,0) * 255).clamp(0,255).byte().numpy()
            sr_viz = (sr_viz.permute(1,2,0) * 255).clamp(0,255).byte().numpy()

            combined = np.vstack([img, gt_viz, sr_viz])
            Image.fromarray(combined).save(os.path.join(save_dir, f"sample_{idx+1}.png"))

    def ensure_csv_has_header(self, csv_path, header):
        # If file does not exist → create and write header
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            return

        # If file exists → read first line
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()

        # If header missing → prepend it
        if first_line != ",".join(header):
            # Read all existing lines
            with open(csv_path, "r") as f:
                existing = f.read()

            # Rewrite with header on first line
            with open(csv_path, "w", newline='') as f:
                f.write(",".join(header) + "\n" + existing)

    def train(self):
        os.makedirs(self.model_path, exist_ok=True)
        unet_path = os.path.join(self.model_path, f'{self.model_type}-{self.dataset}-{self.num_epochs}-{self.lr:.4f}-{self.augmentation_prob:.4f}.pth')

        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
            print(f'{self.model_type} loaded from {unet_path}')
            self.test()
            return

        best_unet_score = -1.0

        for epoch in range(self.num_epochs):
            # Training
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            self.unet.train()
            epoch_loss = 0.0

            AC = SE = SP = PC = F1 = JS = DC = 0.0
            length = 0

            for images, GT in self.train_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)

                # GT is binary 0/1; if empty mask then skip training step
                if GT.sum().item() == 0:
                    continue

                logits = self.unet(images)
                SR = torch.sigmoid(logits)

                loss = self.criterion(logits, GT)
                epoch_loss += loss.item()

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                AC += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

            if length > 0:
                AC /= length; SE /= length; SP /= length; PC /= length
                F1 /= length; JS /= length; DC /= length
            else:
                AC = SE = SP = PC = F1 = JS = DC = None

            print(f'Loss: {epoch_loss:.4f}\n[Training] Acc: {AC}, SE: {SE}, SP: {SP}, PC: {PC}, F1: {F1}, JS: {JS}, DC: {DC}')

            # Validation
            self.unet.eval()
            AC = SE = SP = PC = F1 = JS = DC = 0.0
            length = 0

            with torch.no_grad():
                for images, GT in self.valid_loader:
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    if GT.sum().item() == 0:
                        continue

                    SR = torch.sigmoid(self.unet(images))
                    AC += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length += images.size(0)

            if length > 0:
                AC /= length; SE /= length; SP /= length; PC /= length
                F1 /= length; JS /= length; DC /= length
            else:
                AC = SE = SP = PC = F1 = JS = DC = None

            print(f'[Validation] Acc: {AC}, SE: {SE}, SP: {SP}, PC: {PC}, F1: {F1}, JS: {JS}, DC: {DC}')

            # Decay learning rate
            self.scheduler.step()
            print("Current LR:", self.optimizer.param_groups[0]["lr"])

            unet_score = (JS if JS is not None else 0.0) + (DC if DC is not None else 0.0)
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                torch.save(self.unet.state_dict(), unet_path)
                print(f"Saved best model (score={best_unet_score:.4f}) to {unet_path}")

        self.test()

    def test(self):
        os.makedirs(self.model_path, exist_ok=True)
        unet_path = os.path.join(self.model_path, f'{self.model_type}-{self.dataset}-{self.num_epochs}-{self.lr:.4f}-{self.augmentation_prob:.4f}.pth')

        # Test (load best model)
        self.unet.load_state_dict(torch.load(unet_path))
        self.unet.eval()

        AC = SE = SP = PC = F1 = JS = DC = 0.0
        length = 0

        with torch.no_grad():
            for images, GT in self.test_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)

                if GT.sum().item() == 0:
                    continue

                SR = torch.sigmoid(self.unet(images))
                AC += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

        if length > 0:
            AC /= length; SE /= length; SP /= length; PC /= length
            F1 /= length; JS /= length; DC /= length
        else:
            AC = SE = SP = PC = F1 = JS = DC = None

        # save sample outputs
        save_dir = os.path.join(self.model_path, f"{self.model_type}-{self.dataset}-{self.num_epochs}-{self.lr:.4f}-{self.augmentation_prob:.4f}-samples")
        self.save_samples(save_dir, num_samples=5)

        os.makedirs(self.result_path, exist_ok=True)
        csv_path = os.path.join(self.result_path, 'result.csv')

        # Ensure header exists
        header = ["Model", "Dataset", "LR", "AugProb", "AC", "SE", "SP", "PC", "F1", "JS", "DC"]
        self.ensure_csv_has_header(csv_path, header)

        # Append results
        with open(csv_path, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([self.model_type, self.dataset, self.lr, self.augmentation_prob,
                        AC, SE, SP, PC, F1, JS, DC])
