from torch import nn
from resnet1d import BaseNet
import torch

def sleep_model(n_channels, input_size_samples, n_dim = 256, epoch_len = 7):
    class attention(nn.Module):
        
        def __init__(self, n_dim):
            super(attention,self).__init__()
            self.att_dim = n_dim
            self.W = nn.Parameter(torch.randn(n_dim, self.att_dim))
            self.V = nn.Parameter(torch.randn(self.att_dim, 1))
            self.scale = self.att_dim**-0.5
            
        def forward(self,x):
            x = x.permute(0, 2, 1)
            e = torch.matmul(x, self.W)
            e = torch.matmul(torch.tanh(e), self.V)
            e = e*self.scale
            n1 = torch.exp(e)
            n2 = torch.sum(torch.exp(e), 1, keepdim=True)
            alpha = torch.div(n1, n2)
            x = torch.sum(torch.mul(alpha, x), 1)
            return x
        
    class encoder(nn.Module):

        def __init__(self, n_channels, n_dim):
            super(encoder,self).__init__()
            self.model = BaseNet(input_channel = n_channels)
            self.attention = attention(n_dim)
            
        def forward(self, x): 
            x = self.model(x)
            x = self.attention(x)
            return x
        
    class Net(nn.Module):
        
        def __init__(self, n_channels, n_dim):
            super().__init__()
            self.enc = encoder(n_channels, n_dim)
            self.n_dim = n_dim
            
            self.p1 = nn.Sequential(
                nn.Linear(self.n_dim, self.n_dim // 2, bias=True),
                # nn.BatchNorm1d(self.n_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_dim // 2, self.n_dim // 2, bias=True),
            )
           
        def forward(self, x, proj='mid'):
            x = self.enc(x)
            
            if proj == 'top':
                x = self.p1(x)
                return x
            elif proj == 'mid':
                return x
            else:
                raise Exception("Fix the projection heads")
            
    class final_model(nn.Module):
        
        def __init__(self, n_channels, n_dim, epoch_len):
            super(final_model,self).__init__()
            
            self.epoch_len = epoch_len
            self.net = Net(n_channels, n_dim)
            self.linear_list = nn.ModuleList()
            for i in range(self.epoch_len):
                self.linear_list.append(nn.Linear(n_dim // 2, n_dim // 2, bias=True))
        
        def forward(self, anc, pos, proj = 'top'):
            if proj == 'top':
                anc = self.net(anc, proj = proj)
                pos_features = []
                for i in range(self.epoch_len):
                    pos_features.append(self.linear_list[i](self.net(pos[:, i], proj= proj)))
                pos_features = torch.stack(pos_features, dim=1)
                
                return anc, pos_features
            elif proj == 'mid':
                return self.net(anc)
            else:
                raise Exception("Fix the projection heads")
            
    return final_model(n_channels, n_dim, epoch_len)