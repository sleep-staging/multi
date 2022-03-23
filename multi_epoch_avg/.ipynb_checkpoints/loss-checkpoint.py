import torch
import torch.nn.functional as F

# Loss function
class loss_fn(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, anc, pos):

        # L2 normalize
        anc = F.normalize(anc, p=2, dim=1)  # B, 128
        pos = F.normalize(pos, p=2, dim=2)  # B, 128

        # Calculate weights
        pos = pos.mean(dim=1)
        pos = torch.einsum('ab,cb->ac',anc,pos)
        pos = torch.exp(pos/self.T) # B,B

        # Contrastive loss
        mask = torch.eye(pos.shape[0],device= self.device).bool()
        pos_num = pos.masked_select(mask).view(pos.shape[0],-1)
        loss = -torch.log(pos_num/(pos.sum(axis=-1)))
        
        return loss.mean()

       