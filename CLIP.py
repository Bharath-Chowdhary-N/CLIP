import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Functional as F

class image_encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048,embed_dim)
    def forward(self, x):
        return self.resnet(x)

class text_encoder(nn.Module):
    def __init__(self, vocab_size=50, context_length=64, transformer_width=128, transformer_heads=4, transformer_layers = 6, mlp_ratio=4, embed_dim = 128):
        super().__init__()
        self.vocab_size = vocab_size # We can design only 50 vocab size because we are going to only give in parameters and some text and also some special characters
        self.context_length = context_length # set to 64, as we are gong to process at max 10 params, we can increase this value in future based on need
        self.transformer_width = transformer_width # internal feature size of the transformer, represents each token's hidden representation
        self.transformer_layers = transformer_layers # how many transformer blocks you want to connect in sequence
        self.transformer_heads = transformer_heads # heads for multihead attention
        self.mlp_ratio = mlp_ratio # how much you want to expand the representation of nn final layer
        self.embed_dim = embed_dim #how much you want to project the final text embedding
        # add token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))

        transformer_blocks = [] #initialize empty list

        for ite in range(self.transformer_layers):
            transformer_blocks.append(TransformerBlock(d_model=self.transformer_width, heads=self.transformer_heads, mlp_ratio=self.mlp_ratio))
        
        self.transformer = nn.Sequential(*transformer_blocks) #make sequential transformer block

        self.ln_final = nn.LayerNorm(self.transformer_width) #final layer normalization

        self.text_projection = nn.Parameter(torch.epty(self.transformer_width, self.embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init_normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std = self.transformer_width**-0.5)
    def forward(self, text):
        
        x = self.token_embedding(text) #[batch_size, seq_len, tranformer_width]

        x = x + self.positional_embedding #[batch_size, seq_len, tranformer_width]

        x = self.transformer(x) #[batch_size, seq_len, tranformer_width]

        x = self.ln_final(x) #[batch_size, seq_len, tranformer_width]

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] #take features from end of text embedding, as it would have absorbed features from other embeddings, #[batch_size, transformer_width]

        x = x @ self.text_projection #[batch_size, embed_dim]

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, mlp_ratio=4):
        super().__init__()
        self.attn = MultiheadAttention(d_model, heads)
        self.ln_1 = nn.LayerNorm(d_model) # first layer norm
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model*mlp_ratio), d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) # do prelayer norm, then attn, then residual connection
        x = x + self.mlp(self.ln_2(x)) # same as above, but with mlp
        return x
class MultiheadAttention(nn.Module):
    def __init__(self,d_model, heads):
        super().__init__()
        self.d_model = d_model #d_model as mentioned in paper
        self.heads = heads # num transformer heads
        self.scale = (self.d_model/self.heads)**-0.5 # the denominator sq.(dv) in paper is modified as scale

        #combine the Q,K,V into a single projection
        self.qkv = nn.linear(self.d_model,self.d_model*3)

        #Add a final projection layer
        self.proj = nn.Linear(self.d_model, self.d_model)

    def forward(self,x):
        batch_size, seq_len, _ = x.shape #batch_size is the number of samples to be prcoessed at once, seq_len is the length fo sequence, should match context length
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.heads, -1).transpose(1,2), qkv)
        
        # Compute attention
        attn = (q @ k.tranpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        #apply attention
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.width)
        x = self.proj(x)

        return x

class CLIP(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
    def forward(x):
        pass


