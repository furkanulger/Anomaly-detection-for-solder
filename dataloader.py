from torchvision import datasets, transforms
import torch


class DataLoader:
    def __init__(self, batchSize):
        self.batchSize = batchSize

    def dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('dataset/train/', transforms.Compose([
                transforms.Resize((64, 64)),  # width, height
                transforms.ToTensor()
            ])), batch_size=self.batchSize, num_workers=0, shuffle=True)
        # It only makes sense to apply preprocessing if you have a reason to believe that
        # different input features have different scales (or units),
        # but they should be of approximately equal importance to the learning algorithm.

        valid_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('dataset/validation/', transforms.Compose([
                transforms.Resize((64, 64)),  # width, height
                transforms.ToTensor()
            ])), batch_size=self.batchSize, num_workers=0, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('dataset/test/', transforms.Compose([
                transforms.Resize((64, 64)),  # width, height
                transforms.ToTensor()
            ])), batch_size=self.batchSize, num_workers=0, shuffle=True)

        return train_loader, valid_loader, test_loader
