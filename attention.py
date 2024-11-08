import torch 
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self , nHeads, dEmbed, inProjBias = True , outProjBias=True):
        super().__init__()
        self.inProj = nn.Linear(dEmbed , 3*dEmbed, bias = inProjBias)
        self.outProj = nn.Linear(dEmbed, dEmbed, bias = outProjBias)
        self.nHeads = nHeads
        self.dHead = dEmbed // nHeads

    def forward(self , x , causualMask = False):
        
        inputShape = x.shape
        batchSize , sequenceLength , dEmbed = inputShape
        interimShape = (batchSize,sequenceLength,self.nHeads,self.dHead)
        q,k,v = self.inProj(x).chunk(3,dim=-1)

        q= q.view(interimShape).transpose(1,2)
        k= k.view(interimShape).transpose(1,2)
        v = k.view(interimShape).transoose(1,2)

        weight = q @ k.transpose(-1,-2)

        if causualMask :
            mask = torch.ones_like(weight , dtype=torch.bool).triu(1)
            weight.masked_fill_(mask , -torch.inf)

        weight /= math.sqrt(self.dHead)
        weight = F.softmax(weight , dim =-1)
        output = weight @ v
        output = output.transpose(1,2)
        output = output.reshape(inputShape)
        output = self.outProj(output)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self ,nHeads, dEmbed , dCross , inProjBias = True , outProjBias = True):
        super.__init__()
        self.qProj = nn.Linear(dEmbed, dEmbed, bias = inProjBias)
        self.kProj = nn.Linear(dCross , dEmbed, bias = inProjBias)
        self.vProj = nn.Linear(dCross , dEmbed, bias = inProjBias)
        self.outProj = nn.Linear(dEmbed, dEmbed , bias = outProjBias)
        self.nHeads = nHeads
        self.dHeads = dEmbed // nHeads
    
    def forward(self , x , y):

        inputShape = x.shape
        batchSize, sequenceLength, dEmbed = inputShape
        interimShape = (batchSize, -1 , self.nHeads, self.dHeads)

        q = self.qProj(x)
        k = self.kProj(y)
        v = self.vProj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interimShape).transpose(1,2)
        k = k.view(interimShape).transpose(1,2)
        v = v.view(interimShape).transpose(1,2)


        weight = q @k.transpose(-1,-2)
        weight /= math.sqrt(self.dHeads)
        weight = F.softmax(weight , dim=-1)

        output = weight @v
        output = output.transpose(1,2).contigous()
        output = output.view(inputShape)
        output = self.outProj(output)
        return output

