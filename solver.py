import os
import numpy as np
import time
import datetime
import torch
import torchvision
from PIL import Image
from torch import optim
from torch.autograd import Variable
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
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path

        self.mode = config.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
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

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        print(model)
        print(name)
        print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters())))

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
            img = images[i].cpu()              # (3, H, W)
            gt = GT[i].cpu()                   # (1, H, W)
            sr = SR[i].cpu()                   # (1, H, W)

            # Make sure gt and sr become 3-channel
            if gt.size(0) == 1:
                gt = gt.repeat(3, 1, 1)
            if sr.size(0) == 1:
                sr = sr.repeat(3, 1, 1)

            # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0)
            gt  = gt.permute(1, 2, 0)
            sr  = sr.permute(1, 2, 0)

            # Normalize to 0–255 uint8
            img = (img * 255).byte().numpy()
            gt  = (gt  * 255).byte().numpy()
            sr  = (sr  * 255).byte().numpy()

            # vertical stacking
            combined = np.vstack([img, gt, sr])

            Image.fromarray(combined).save(f"{save_dir}/sample_{i+1}.png")


    def train(self):
        #====================================== Training ===========================================#
        
        os.makedirs(self.model_path, exist_ok=True)
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.0

            for epoch in range(self.num_epochs):
                print('Epoch [%d/%d]' % (epoch+1, self.num_epochs))
                self.unet.train(True)
                epoch_loss = 0
                
                acc = 0.0       # Accuracy
                SE = 0.0        # Sensitivity (Recall)
                SP = 0.0        # Specificity
                PC = 0.0        # Precision
                F1 = 0.0        # F1 Score
                JS = 0.0        # Jaccard Similarity
                DC = 0.0        # Dice Coefficient
                length = 0

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    has_gt = GT.sum() > 0   # GT is non-empty only if real mask exists

                    if has_gt:
                        # SR : Segmentation Result
                        SR = torch.sigmoid(self.unet(images))
                        SR_flat = SR.view(SR.size(0), -1)
                        GT_flat = GT.view(GT.size(0), -1)
                        loss = self.criterion(SR_flat, GT_flat)
                        epoch_loss += loss.item()

                        # Backprop + optimize
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

                if length == 0:
                    acc = SE = SP = PC = F1 = JS = DC = None
                else:
                    acc /= length
                    SE /= length
                    SP /= length
                    PC /= length
                    F1 /= length
                    JS /= length
                    DC /= length

                # Print the log info
                print('Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                      epoch_loss,\
                      acc, SE, SP, PC, F1, JS, DC))

                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    self.update_lr(lr)
                    print ('Decay learning rate to lr: {}.'.format(lr))


                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.0       # Accuracy
                SE = 0.0        # Sensitivity (Recall)
                SP = 0.0        # Specificity
                PC = 0.0        # Precision
                F1 = 0.0        # F1 Score
                JS = 0.0        # Jaccard Similarity
                DC = 0.0        # Dice Coefficient
                length=0

                for i, (images, GT) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    has_gt = GT.sum() > 0   # GT is non-empty only if real mask exists

                    if has_gt:
                        SR = torch.sigmoid(self.unet(images))
                        acc += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)

                        length += images.size(0)

                if length == 0:
                    acc = SE = SP = PC = F1 = JS = DC = None
                else:
                    acc /= length
                    SE /= length
                    SP /= length
                    PC /= length
                    F1 /= length
                    JS /= length
                    DC /= length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc, SE, SP, PC, F1, JS, DC))
                
                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type, epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type, epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type, epoch+1)))
                '''


                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f'%(self.model_type, best_unet_score))
                    torch.save(best_unet, unet_path)
    
            #===================================== Test ====================================#
            del self.unet
            del best_unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))
            
            self.unet.train(False)
            self.unet.eval()

            acc = 0.0       # Accuracy
            SE = 0.0        # Sensitivity (Recall)
            SP = 0.0        # Specificity
            PC = 0.0        # Precision
            F1 = 0.0        # F1 Score
            JS = 0.0        # Jaccard Similarity
            DC = 0.0        # Dice Coefficient
            length=0
            for i, (images, GT) in enumerate(self.test_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)

                has_gt = GT.sum() > 0   # GT is non-empty only if real mask exists

                if has_gt:
                    SR = torch.sigmoid(self.unet(images))
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                            
                    length += images.size(0)
            
            if length == 0:
                acc = SE = SP = PC = F1 = JS = DC = None
            else:
                acc /= length
                SE /= length
                SP /= length
                PC /= length
                F1 /= length
                JS /= length
                DC /= length
            unet_score = JS + DC

            save_dir = os.path.join(
                self.model_path,
                '%s-%d-%.4f-%d-%.4f' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob)
            )
            self.save_samples(save_dir)
            os.makedirs(self.result_path, exist_ok=True)
            f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, best_epoch, self.num_epochs, self.num_epochs_decay, self.augmentation_prob])
            f.close()
