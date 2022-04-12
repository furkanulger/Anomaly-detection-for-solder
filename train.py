import argparse
import os

import numpy as np
import torch.utils.data
from torch import nn, optim
from torchvision.utils import save_image

from CAE import CAE
from betaVAE import betaVAE
from dataloader import DataLoader

# argument parser ile command line script'e input ver.
parser = argparse.ArgumentParser(description="Model training arguments")
parser.add_argument("-m", "--model", required=True, help="The model to be trained")
parser.add_argument("-b", "--beta", help=" KL weight for beta-VAE", default=1)
parser.add_argument("-bs", "--batchsize", help="Batch size for the model", default=8)
parser.add_argument("-lr", "--lrnrate",  help="Learning rate for the model", default=1e-4)
parser.add_argument("-e", "--numEpochs", help="Number of epochs for the model", default=100)
args = parser.parse_args()
print(args)


class train_model():

    def train(self):
        sample_interval = 100

        save_path_model = r".\Results\Model_checkpoints"
        save_recons = r".\Results\training_Reconstructions"

        os.makedirs("./dataset/train", exist_ok=True)
        os.makedirs("./dataset/validation", exist_ok=True)
        os.makedirs("./dataset/test", exist_ok=True)
        os.makedirs(save_path_model, exist_ok=True)
        os.makedirs(save_recons, exist_ok=True)

        if args.model == "CAE":
            model = CAE().cuda()
            print("Training CAE")
            print("Number of epochs: {0}".format(args.numEpochs))
        elif args.model == "VAE":
            model = betaVAE().cuda()  # send model to the default GPU device.
            print("Training VAE")
            print("Number of epochs: {0}, KL weight: {1}".format(args.numEpochs, args.beta))
        else:
            raise ValueError("The model should be either 'CAE' or 'VAE'")

        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=float(args.lrnrate), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

        valid_loss_min = np.Inf
        Loader = DataLoader(int(args.batchsize))
        train_loader, valid_loader, _ = Loader.dataloader()
        batches_done = 0
        for epoch in range(int(args.numEpochs)):
            train_loss = 0.0
            valid_loss = 0.0
            valid_recons_loss = 0
            for data in train_loader:  # there is no label.
                img, _ = data
                optimizer.zero_grad()
                # ===================forward pass=====================
                if args.model == "CAE":
                    # img= img.view(-1, 64*64*3) # Bx3x64x64
                    img = img.cuda()
                    x_reconst = model(img)
                elif args.model == "VAE":
                    img = img.cuda()
                    x_reconst, mu, log_var = model(img)  # send images to gpu then to the model
                    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # Compute reconstruction loss and kl divergence
                # For KL divergence, see Appendix B in VAE paper [Kingma, 2014]
                recons_loss = criterion(x_reconst, img)
                if args.model == "CAE":
                    loss = recons_loss
                elif args.model == "VAE":
                    loss = recons_loss + (int(args.beta) * kl_div)

                # ===================backpropagation====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ===================log========================
                # update running training loss
                train_loss += loss.item() * img.size(0)  # loss * 8 (batch_Size)

                # SAVE IMAGES
                x_save = x_reconst.data[:25]
                if batches_done % sample_interval == 0:
                    save_image(x_save.view(-1, 3, 64, 64), save_recons +"/%d.png" % batches_done,
                               nrow=5,
                               normalize=False)
                batches_done += 1
                ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            for img, _ in valid_loader:
                img = img.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                if args.model == "CAE":
                    x_reconst = model(img)
                elif args.model == "VAE":
                    x_reconst, mu, log_var = model(img)  # send images to gpu then to the model
                    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # calculate the loss

                recons_loss = criterion(x_reconst, img)
                if args.model == "CAE":
                    loss = recons_loss
                elif args.model == "VAE":
                    loss = recons_loss + (int(args.beta) * kl_div)

                # update running validation loss
                valid_loss += loss.item() * img.size(0)
                # gives loss of the entire mini-batch, but divided by mini-batch size due to taking mean.
                valid_recons_loss += recons_loss.item() * img.size(0)

            ## print training/validation statistics
            ## calculate average loss over an epoch by dividing to the whole dataset
            valid_loss = valid_loss / len(valid_loader)  # divide by number of batches to get avg. batch loss
            train_loss = train_loss / len(train_loader)
            valid_recons_loss = valid_recons_loss / len(valid_loader)
            print('Epoch: {} \tTraining Loss: {:.4f} \t Validation loss: {:.4f}'.format(epoch, train_loss, valid_loss))
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), save_path_model + "\\model_lowest_val_loss.pt")
                valid_loss_min = valid_loss
                print(
                    "Avg.Reconstruction loss per sample in validation set (normal samples) for anomaly score: {:.3f}".format(
                        valid_recons_loss))
            scheduler.step(valid_loss)  # for reduceonplateu


train_model().train()
