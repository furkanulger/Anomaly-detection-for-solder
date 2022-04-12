import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from dataloader import DataLoader
from betaVAE import betaVAE
from CAE import CAE
import os
import argparse

"""Tests CAE or VAE on the test set using Gradient Constraint (GradCon) as an anomaly score. Implementation for 
GradCon is from https://github.com/olivesgatech/gradcon-anomaly """

# Command line inputs for the script
parser = argparse.ArgumentParser(description="Model testing arguments")
parser.add_argument("-m", "--model", required=True, help="The model to be tested (either CAE or VAE)")
parser.add_argument("-p", "--path", required=True, help="Path for the model to be tested")
parser.add_argument("-th", "--threshold", required=True, help="Threshold for classification")
parser.add_argument("-b", "--beta", help=" KL weight for beta-VAE", default=1)
parser.add_argument("-t", "--numTest", help="Number of tests", default=1)
args = parser.parse_args()
print(args)


class test_model:
    def __init__(self):
        self.save_path = ".\Results\test_Reconstructions"
        self.grad_loss_weight = 0.03 * 4  # This is the suggested parameter by [Kwon et al.], see https://github.com/olivesgatech/gradcon-anomaly
        self.num_decoder_layers = 6

    def test(self):

        os.makedirs("./dataset/train", exist_ok=True)
        os.makedirs("./dataset/validation", exist_ok=True)
        os.makedirs("./dataset/test", exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

        if args.model == "CAE":
            model = CAE().cuda()
            print("Testing Convolutional Autoencoder")
        elif args.model == "VAE":
            model = betaVAE().cuda()
            print("Testing VAE")
        else:
            raise ValueError("The model should be either 'CAE' or 'VAE'")

        model = torch.nn.DataParallel(model).cuda()
        model.eval()  # to set dropout and batch normalization layers to evaluation mode before running inference

        checkpoint_model = torch.load(args.path)
        model.load_state_dict(checkpoint_model['state_dict'])
        ref_grad = checkpoint_model['ref_grad']

        criterion = nn.MSELoss(reduction='mean')  # average over the batch (loss for a sample)

        sum_precision = []
        sum_recall = []
        sum_f1_score = []
        avg_rec_error = 0

        for test in range(int(args.numTest)):

            _, _, test_loader = DataLoader(batchSize=1).dataloader()
            print("Number of samples in the test set:", len(test_loader))

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            sum_rec_error = 0

            for num, data in enumerate(test_loader):
                img, label = data  # label 0: Defective , 1 is intact.
                label_np = label.numpy()
                img = img.cuda()  # .to(device)#
                model.zero_grad()  # not to sum gradients
                if args.model == "CAE":
                    output = model(img)
                elif args.model == "VAE":
                    output, mu, log_var = model(img)

                mse_loss = criterion(output, img)
                recons_loss = mse_loss
                mse_loss.backward()  # computes dL/dW

                grad_loss = 0
                for i in range(self.num_decoder_layers):
                    if i <= 2:  # for indexes of the transposed conv. layers
                        target_grad = model.module.decoder[int(3 * i)].weight.grad  # conv. layer weights
                    else:
                        target_grad = model.module.decoder[int(4 * i - 2)].weight.grad  # conv. layer weights

                    grad_loss += F.cosine_similarity(target_grad.view(-1, 1),
                                                     ref_grad[i].avg.view(-1, 1),
                                                     dim=0)  # reference grad. = extracted grad. from training data
                    # and target grad weights are compared

                grad_loss = grad_loss / self.num_decoder_layers

                anomaly_score = -1 * recons_loss + self.grad_loss_weight * grad_loss
                anomaly_score = anomaly_score.cpu().detach().numpy()
                sum_rec_error += anomaly_score
                predicted = 1 if anomaly_score >= float(args.threshold) else 0

                # class 0: defective class 1: intact.
                if predicted == 0 and label_np == predicted: tp += 1
                if predicted == 0 and label_np != predicted: fp += 1
                if predicted == 1 and label_np == predicted: tn += 1
                if predicted == 1 and label_np != predicted: fn += 1

            ####RESULTS
            avg_rec_error += sum_rec_error / len(test_loader)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)  # f1 score
            sum_precision.append(precision)
            sum_recall.append(recall)
            sum_f1_score.append(f1)

        std_precision = np.std(sum_precision)
        std_recall = np.std(sum_recall)
        std_f1score = np.std(sum_f1_score)

        print("Average loss on the test set is:", avg_rec_error / int(args.numTest))
        print(
            "precision: {0:.3f} ± {3:.3f}, recall= {1:.3f} ± {4:.3f}, f1 score= {2:.3f} ± {5:.3f} on average of {6:d} "
            "runs.".format(
                np.sum(sum_precision) / int(args.numTest), np.sum(sum_recall) / int(args.numTest),
                np.sum(sum_f1_score) / int(args.numTest), std_precision,
                std_recall, std_f1score, int(args.numTest)))


test_model().test()
