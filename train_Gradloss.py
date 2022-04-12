""" Implementation for training with gradient constraint is from:
https://github.com/olivesgatech/gradcon-anomaly"""

from betaVAE import betaVAE
from CAE import CAE
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from dataloader import DataLoader
import argparse
from utils import AverageMeter
import os

# argument parser ile command line script'e input ver.
parser = argparse.ArgumentParser(description="Model training arguments")
parser.add_argument("-m", "--model", required=True, help="The model to be trained")
parser.add_argument("-b", "--beta", help=" KL weight for beta-VAE", default=1)
parser.add_argument("-bs", "--batchsize", help="Batch size for the model", default=8)
parser.add_argument("-lr", "--lrnrate",  help="Learning rate for the model", default=1e-4)
parser.add_argument("-e", "--numEpochs", help="Number of epochs for the model", default=100)
args = parser.parse_args()
print(args)



class train_gradcon():
    def __init__(self):
        self.num_decoder_layers = 6
        self.grad_loss_weight = 0.03

    def train(self):

        save_path = ".\Results\Model_checkpoints"
        os.makedirs("./dataset/train", exist_ok=True)
        os.makedirs("./dataset/validation", exist_ok=True)
        os.makedirs("./dataset/test", exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        fullpath = save_path + "\checkpoint_minVal.pth.tar"

        if args.model == "CAE":
            model = CAE().cuda()
            print("Training CAE")
        elif args.model == "VAE":
            model = betaVAE().cuda()  # send model to the default GPU device.
            print("Training VAE")
            print("Number of epochs: {0}, kl weight: {1}".format(int(args.numEpochs), int(args.beta)))
        else:
            raise ValueError("The model should be either 'CAE' or 'VAE'")

        model = torch.nn.DataParallel(model).cuda()

        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=float(args.lrnrate), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

        # Keep track of traning gradients to calculate the gradient loss
        ref_grad = []
        for i in range(self.num_decoder_layers):
            layer_grad = AverageMeter()
            if i <= 2:  # for indexes of the transposed conv. layers
                layer_grad.avg = torch.zeros(model.module.decoder[int(3 * i)].weight.shape).cuda()
            else:
                layer_grad.avg = torch.zeros(model.module.decoder[int(4 * i - 2)].weight.shape).cuda()
            ref_grad.append(layer_grad)

        train_loader, valid_loader, test_loader = DataLoader(batchSize=int(args.batchsize)).dataloader()
        valid_loss_min = np.Inf
        valid_loss_values = []

        for epoch in range(int(args.numEpochs)):

            train_losses = AverageMeter()
            recon_losses = AverageMeter()
            grad_losses = AverageMeter()
            for data in train_loader:  # there is no label.
                img, _ = data
                img = img.cuda()
                optimizer.zero_grad()
                # ===================forward pass=====================
                if args.model == "VAE":
                    x_reconst, mu, log_var = model(img)  # send images to gpu then to the model
                    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                elif args.model == "CAE":
                    x_reconst = model(img)
                # Compute reconstruction loss and kl divergence
                # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
                mse_loss = criterion(x_reconst, img)

                # kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                # -5e-4
                # Calculate the gradient loss for each layer
                grad_loss = 0
                for i in range(self.num_decoder_layers):
                    if i <= 2:  # for indexes of the transposed conv. layers
                        wrt = model.module.decoder[int(3 * i)].weight  # conv. layer weights
                    else:
                        wrt = model.module.decoder[int(4 * i - 2)].weight  # conv. layer weights
                    target_grad = torch.autograd.grad(mse_loss, wrt, create_graph=True, retain_graph=True)[0]

                    grad_loss += -1 * F.cosine_similarity(target_grad.view(-1, 1),
                                                          ref_grad[i].avg.view(-1, 1), dim=0)

                # In the first iteration, since there is no history of training gradients, gradient loss is not utilized
                if ref_grad[0].count == 0:
                    grad_loss = torch.FloatTensor([0.0]).cuda()
                else:
                    grad_loss = grad_loss / self.num_decoder_layers
                if args.model == "VAE":
                    loss = mse_loss + self.grad_loss_weight * grad_loss + (kl_div * int(args.beta))
                elif args.model == "CAE":
                    loss = mse_loss + self.grad_loss_weight * grad_loss
                train_losses.update(loss.item(), img.size(0))  # img.size(0): Batch size
                recon_losses.update(mse_loss.item(), img.size(0))
                grad_losses.update(grad_loss.item(), img.size(0))
                loss.backward()

                # Update the reference gradient
                for i in range(self.num_decoder_layers):
                    if i <= 2:  # for indexes of the transposed conv. layers
                        ref_grad[i].update(model.module.decoder[3 * i].weight.grad)
                    else:
                        ref_grad[i].update(model.module.decoder[4 * i - 2].weight.grad)

                optimizer.step()
            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            valid_losses = AverageMeter()
            valid_recons = AverageMeter()
            valid_grad = AverageMeter()
            for img, _ in valid_loader:
                img = img.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                if args.model == "CAE":
                    x_reconst = model(img)
                elif args.model == "VAE":
                    x_reconst, mu, log_var = model(img)  # send images to gpu then to the model
                    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # calculate the loss
                mse_loss = criterion(x_reconst, img)

                # Calculate the gradient loss for each layer
                grad_loss = 0
                for i in range(self.num_decoder_layers):
                    if i <= 2:  # for indexes of the transposed conv. layers
                        wrt = model.module.decoder[int(3 * i)].weight  # conv. layer weights
                    else:
                        wrt = model.module.decoder[int(4 * i - 2)].weight  # conv. layer weights
                    target_grad = torch.autograd.grad(mse_loss, wrt, create_graph=True, retain_graph=True)[0]
                    # dL / dW
                    grad_loss += -1 * F.cosine_similarity(target_grad.view(-1, 1),
                                                          ref_grad[i].avg.view(-1, 1), dim=0)

                # In the first iteration, since there is no history of training gradients, gradient loss is not utilized
                if ref_grad[0].count == 0:
                    grad_loss = torch.FloatTensor([0.0]).cuda()
                else:
                    grad_loss = grad_loss / self.num_decoder_layers
                if args.model == "CAE":
                    loss = mse_loss + self.grad_loss_weight * grad_loss
                elif args.model == "VAE":
                    loss = mse_loss + self.grad_loss_weight * grad_loss + (kl_div * int(args.beta))

                valid_losses.update(loss.item(), img.size(0))  # img.size(0): Batch size
                valid_recons.update(mse_loss.item(), img.size(0))
                valid_grad.update(grad_loss.item(), img.size(0))


            valid_loss_values.append(valid_losses.avg)
            print('Epoch: {} \t'
                  'Training Loss: {:.4f} \t'
                  'Validation loss: {:.4f}'.format(epoch, train_losses.avg, valid_losses.avg))

            # save model if validation loss has decreased
            if valid_losses.avg <= valid_loss_min:
                print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_losses.avg))
                valid_loss_min = valid_losses.avg
                torch.save(model.state_dict(), fullpath)

            scheduler.step(valid_losses.avg)  # for reduceonplateu

        print("Variance of the validation loss is:", np.var(valid_loss_values))
        print("Proposed anomaly score for the test set is:", valid_loss_min + np.var(valid_loss_values))

train_gradcon().train()