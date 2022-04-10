
""" convolutional autoencoder model architecture"""

from torch import nn


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
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
            #nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride= 1, padding=1),  # 64x64        
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),    
            nn.Conv2d(256, 10, 3, stride= 1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),    
        )
     
        self.decoder = nn.Sequential(

          nn.ConvTranspose2d(10, 256, 3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(),  
       
          #nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
          #nn.BatchNorm2d(256),
          #nn.LeakyReLU(),  
          nn.ConvTranspose2d(256, 128, 3, stride= 1, padding=1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(),  

          #nn.Upsample(scale_factor= 2, mode= 'bicubic'),
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
          nn.BatchNorm2d(3),
          nn.Tanh(), 
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recons = self.decoder(z)
        return x_recons



