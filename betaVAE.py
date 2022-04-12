""" beta-VAE model architecture"""

import torch
import torch.utils.data
from torch import nn

torch.manual_seed(0)

class betaVAE(nn.Module):
    def __init__(self):
        super(betaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride= 1,padding=1),    # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),    
            nn.MaxPool2d(2, stride=2), 

            nn.Conv2d(16,32, 3, stride= 1, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),    
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            
            nn.Conv2d(32,64, 3, stride= 1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),        
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride= 1,padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 3, stride= 1, padding=1),  # 64x64        
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        # the gaussian distribution of each feature with 2 variables, mean and variance.
        self.mu_encoder = nn.Sequential(
            nn.Conv2d(256, 10, 3, stride= 1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),  
        ) 
        self.logvar_encoder = nn.Sequential(
            nn.Conv2d(256, 10, 3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),  
        ) 

        self.decoder = nn.Sequential(

          nn.ConvTranspose2d(10, 256, 3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(),  

          nn.ConvTranspose2d(256, 128, 3, stride= 1, padding=1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(),  


          nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(),

          nn.Upsample(scale_factor= 2, mode= 'bicubic'),
          nn.ConvTranspose2d(64, 32, 3, stride= 1, padding=1),
          nn.BatchNorm2d(32),
          nn.LeakyReLU(),  

          nn.Upsample(scale_factor= 2, mode= 'bicubic'),
          nn.ConvTranspose2d(32,16, 3, stride=1, padding=1),
          nn.BatchNorm2d(16),
          nn.LeakyReLU(),  

          nn.Upsample(scale_factor= 2, mode= 'bicubic'),
          nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
          nn.BatchNorm2d(3), # BU OLMADIĞINDA ANOMALİ GÖRÜNTÜLER DAHA DÜZGÜN RECONSTRUCT EDİLİYOR.
          nn.Tanh(), 
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu= self.mu_encoder(x)
        log_var= self.logvar_encoder(x)
        mu= mu.view(-1, 10*8*8)
        log_var= log_var.view(-1, 10*8*8)

        z= self.reparameterize(mu, log_var)
        z= z.reshape(-1, 10,8,8)
        x_recons = self.decoder(z)

        return x_recons, mu, log_var


