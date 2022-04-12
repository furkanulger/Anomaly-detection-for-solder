import numpy as np
import torch.utils.data
from torch import nn
from betaVAE import betaVAE
from CAE import CAE
from torchvision.utils import save_image
from dataloader import DataLoader
import argparse
import os

""" Tests CAE or VAE on the test set using either reconstruction loss or ELBO as an anomaly score"""

# argument parser ile command line script'e input ver.
parser = argparse.ArgumentParser(description="Model testing arguments")
parser.add_argument("-m", "--model", required=True, help="The model to be tested (either CAE or VAE)")
parser.add_argument("-p", "--path", required=True, help="Path for the model to be tested")
parser.add_argument("-a", "--score", required=True, help="Anomaly score for testing (either ELBO or Recon)")
parser.add_argument("-th", "--threshold", required=True, help="Threshold for classification")
parser.add_argument("-b", "--beta", help=" KL weight for beta-VAE", default= 1)
parser.add_argument("-t", "--numTest", help="Number of tests", default= 1)
args = parser.parse_args()
print(args)


class TestModel:

    def test(self):

        save_path = ".\Results\test_Reconstructions"
        os.makedirs("./dataset/train", exist_ok=True)
        os.makedirs("./dataset/validation", exist_ok=True)
        os.makedirs("./dataset/test", exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        if args.model == "CAE":
            model = CAE().cuda()  # send model to the default GPU device.
            print("Convolutional Autoencoder is being tested with recons. loss as an anomaly score.")
        elif args.model == "VAE":
            model = betaVAE().cuda()
            print("VAE is being tested with ELBO loss as an anomaly score") if (args.score == "ELBO") else print(
                "VAE is being tested with reconstruction loss as an anomaly score")
        else:
            raise ValueError("The model should be either 'CAE' or 'VAE'")

        model.load_state_dict(torch.load(args.path))  # load into the model
        model.eval()  # to set dropout and batch normalization layers to evaluation mode before running inference

        criterion = nn.MSELoss(
            reduction='mean')  # ‘mean’ the loss is summed for every example across every element and then divided by the total amount of examples*elements.

        sum_precision = []
        sum_recall = []
        sum_f1_score = []

        for number_i in range(int(args.numTest)):

            _, _, test_loader = DataLoader(batchSize=1).dataloader()

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            batches_done = 0
            for img, label in test_loader:
                label_np = label.numpy()
                img = img.cuda()
                if args.model == "CAE":
                    output = model(img)
                elif args.model == "VAE":
                    output, mu, log_var = model(img)
                    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                save_image(output.view(-1, 3, 64, 64), save_path +"\\%d.png" % batches_done,
                           nrow=1,
                           normalize=False)
                batches_done += 1

                for j in range(len(img)):
                    mse_loss = criterion(output.cpu()[j, :, :, :], img.cpu()[j, :, :, :])

                    if args.score == "ELBO":
                        loss = -mse_loss + (int(args.beta) * kl_div)
                    else:
                        loss = mse_loss
                    loss = loss.cpu().detach().numpy()
                    predicted = 1 if loss <= float(args.threshold) else 0

                    # class 0: defective class 1: intact.
                    if predicted == 0 and label_np[j] == predicted: tp += 1
                    if predicted == 0 and label_np[j] != predicted: fp += 1
                    if predicted == 1 and label_np[j] == predicted: tn += 1
                    if predicted == 1 and label_np[j] != predicted: fn += 1

            ####RESULTS
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)  # f1 score

            sum_precision.append(precision)
            sum_recall.append(recall)
            sum_f1_score.append(f1_score)

        std_precision = np.std(sum_precision)
        std_recall = np.std(sum_recall)
        std_f1score = np.std(sum_f1_score)

        print(
            "precision: {0:.3f} ± {3:.3f}, recall= {1:.3f} ± {4:.3f}, f1 score= {2:.3f} ± {5:.3f} on average of {6:d} "
            "runs.".format(
                np.sum(sum_precision) / int(args.numTest), np.sum(sum_recall) / int(args.numTest),
                np.sum(sum_f1_score) / int(args.numTest),
                std_precision, std_recall, std_f1score, int(args.numTest)))


TestModel().test()
