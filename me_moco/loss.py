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
        pos = F.normalize(pos, p=2, dim=2)  # B, 7, 128
        
        # Calculate weights
        pos_weights = torch.einsum('ab,dcb->adc',anc,pos)
        pos_weights = self.softmax(pos_weights) # B,B,7
        pos = torch.einsum('adc,dce->ade',pos_weights,pos) # B,B,128
        pos = anc.unsqueeze(1)*pos # B,B,128
        pos = torch.exp(pos.sum(axis=-1)/self.T) # B,B
        
        # neg sampling with the anchor points
        neg_anc = torch.einsum('ab,cb->ac',anc,anc)
        neg_anc = torch.exp(neg_anc/self.T) # B,B
        neg_mask = ~torch.eye(neg_anc.shape[0],device=self.device).bool()
        neg_anc = neg_mask.masked_select(neg_mask).view(neg_mask.shape[0],-1)
        
        # Contrastive loss
        mask = torch.eye(pos.shape[0],device=self.device).bool()
        pos_num = pos.masked_select(mask).view(pos.shape[0],-1)
        denom = torch.cat([pos,neg_anc],axis=-1)
        loss = -torch.log(pos_num/(denom.sum(axis=-1)))
        
        return loss.mean()

