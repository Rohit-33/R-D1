import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self , nVocab : int , nEmbed: int, nToken : int ):
        super().__init__()
        self.tokenEmbedding = nn.Embedding(nVocab , nEmbed)
        self.positionalEmbedding = nn.Parameter(torch.zeros(nToken, nEmbed))

    def forwrad(self , tokens):
        x = self.tokenEmbedding(tokens)
        x += self.positionalEmbedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self , nHeads : int , nEmbed : int):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(nEmbed)
        self.attention = SelfAttention(nHeads , nEmbed)
        self.layernorm2 = nn.LayerNorm(nEmbed)
        self.linear1 = nn.Linear(nEmbed, 4*nEmbed)
        self.linear2 = nn.Linear(4*nEmbed , nEmbed)

    def forward(self , x):
        residue = x
        x = self.layernorm1(x)
        x = self.attention(x , causal_mask = True)
        x += residue

        residue = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x*torch.sigmoid(1.702 *x) # QuickGELU activation function
        x = self.linear2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49407 , 768, 77)
        self.moduleLayers = nn.ModuleList([
            CLIPLayer(12 , 768) for i in range(16)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, token:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layers in self.moduleLayers:
            state = layers(state)
        output = self.layernorm(state)
        return output 
