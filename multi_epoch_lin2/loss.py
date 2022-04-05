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
        pos = torch.cat((pos[:,:3],pos[:,4:]),axis=1)
        
        # Calculate weights
        pos_weights = torch.einsum('ab,dcb->adc',anc,pos)
        pos_weights = self.softmax(pos_weights) # B,B,7
        pos_weights = torch.ones_like(pos_weights) # removing the weightage thing
        pos = torch.einsum('adc,dce->ade',pos_weights,pos) # B,B,128
        pos = anc.unsqueeze(1)*pos # B,B,128
        pos = torch.exp(pos.sum(axis=-1)/self.T) # B,B
        
        # Contrastive loss
        mask = torch.eye(pos.shape[0],device=self.device).bool()
        pos_num = pos.masked_select(mask).view(pos.shape[0],-1)
        loss = -torch.log(pos_num/(pos.sum(axis=-1)))
        
        return loss.mean()
