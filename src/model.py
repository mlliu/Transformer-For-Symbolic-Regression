"""
Transformer_encoder model
and Transformer_decoder model
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

import attention

#one Encoder_block consisits of a self-attention block and a ffd layer
#nn.Module base class for all neural network modeules
class Encoder_Block(nn.Module):
    """ an unassuming Transformer block"""
    def __init__(self,config):
        super().__init__()
        
        #batch normilization over the last dimension why?
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.att = attention.CasualSelfAttention(config)
        
        #position-wise ffd
        self.ffd = nn.Sequential(
            nn.Linear(config.n_embd,4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd,config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self,x):
        #encoder_bloack, causal = none
        #x = x + self.att(self.ln1(x),causal = None)
        #x = x + self.ffd(self.ln2(x))
        x = self.ln1(x + self.att(x ,causal=None))
        x = self.ln2(x + self.ffd(x))
        
        return x
"""
one decoder block consists a masked self-attention, a attention over the encoder of the encoder stack
 and a feed forward 
"""
class Decoder_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        #there are three LayerNorm in the decoder
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        
        #two attention layer, one is masked self-attention
        #                     second is att_encoder
        self.self_masked_att = attention.CasualSelfAttention(config)# causal =True 
        self.att_enconder = attention.CasualSelfAttention(config)   #causal =True attend encoded_input
        
        #position-wise ffd
        self.ffd = nn.Sequential(
            nn.Linear(config.n_embd,4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd,config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
    def forward(self, x ,encoded_input=None):
        # decoder must have the encoded_input
        assert(encoded_input is not None)
        
        x = self.ln1(x + self.self_masked_att(x ,causal=True))
        
        x = self.ln2(x + self.att_enconder(x, causal=True,encoded_input=encoded_input)) 
        
        x = self.ln3(x + self.ffd(x))
        
        return x
 
class TransformerConfig:
    """ base Transformer config, params common to all Transformer model"""
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    n_enc_layer = 6
    n_dec_layer =6
    n_head = 8
    n_embd = 512
    
    assert(n_embd % n_head==0, "embedding dimension must be a multiple of n_head")
    
    def __init__(self, vocab_size,block_size,**kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():   #what does this doing?
            setattr(self,k,v)
            
class Transformer(nn.Module):
    """the full Transformer model, with a context size T of block_size"""
    
    def __init__(self,config):
        super().__init__()
        
        self.n_embd = config.n_embd
        #token embedding layer
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        #torch.nn.parameter.Parameter(data=None, requires_grad=True) requires gradient
        self.input_positional_embedding = nn.Parameter(torch.zeros(1, config.block_size,config.n_embd))
        self.input_ln_embd = nn.LayerNorm(config.n_embd)
        self.input_drop = nn.Dropout(config.embd_pdrop)
        
        #encoder_block
        self.encoder_block = nn.Sequential(*[Encoder_Block(config) for _ in range(config.n_enc_layer)])
        
        
        self.output_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        #torch.nn.parameter.Parameter(data=None, requires_grad=True) requires gradient
        self.output_positional_embedding = nn.Parameter(torch.zeros(1, config.block_size,config.n_embd))
        self.output_ln_embd = nn.LayerNorm(config.n_embd)
        self.output_drop = nn.Dropout(config.embd_pdrop)
        
        #decoder_block
        
        self.decoder_block = nn.Sequential(*[Decoder_Block(config) for _ in range(config.n_dec_layer )])
        
        #decoder head
        
        self.decoder_head = nn.Linear(config.n_embd, config.vocab_size)
        
        self.block_size = config.block_size
        self.apply(self._init_weights)
        
        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))
              
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
          return self.block_size

    #x1 is not an id, is already a (b,t,n_embd) size vector
    #x1,x2,y = x2[1:].mask_select()
    #x1 is dataset, x2 is pad+function, we can get y form x2 
    def forward(self,x1,x2_id, target=None ):

        x1_b, x1_len =x1.size() # (b,t)
        assert x1_len == self.n_embd * self.block_size #8*16 =128
        x1_t = self.block_size
        x1 = x1.view(x1_b,-1,self.n_embd)
        #assert in_t <= self.block_size,"cannot forward, model block size is exhuasted"
        #assert x1_embd = self.n_embd #其实不相等，也可以在做矩阵@的时候补齐

        #encoder
        #in_x = self.input_embedding(in_id) #(b,t c)==(b,t,n_embd)
        x1_position =self.input_positional_embedding[:,:x1_t,:] #each position maps to a learnable vector
        in_x = self.input_ln_embd(x1+x1_position)
        in_x = self.input_drop(in_x)
        encoded_input = self.encoder_block(in_x) #serve as


        #decoder
        x2_b, x2_t =x2_id.size()
        print(x2_id)
        print(x2_t)
        assert x2_t <= self.block_size, "cannot forward, model block size is exhuasted"

        out_x2 = self.output_embedding(x2_id)
        out_position = self.output_positional_embedding[:,:x2_t,:]
        out_x2 = self.output_ln_embd(out_x2+out_position)
        out_x2 = self.output_drop(out_x2)
        
        print(encoded_input)
        out_x2 = self.decoder_block(out_x2,encoded_input)  #(B,T,C)

        logits = self.decoder_head(out_x2) # (b, t, n_embd) * (n_embd,n_vocab) = (b,t, n_vocab)

        loss = 0
        #if we are given some desired targets also calculate the loss
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), #(b*t, n_vocab)
                                  targets.view(-1),ignore_index=0)   #(b*t)

        return logits,loss    



    """  
    #input size(b,t), target size(b,t)
    def forward(self,in_id,out_id, target=None ):

        in_b, in_t =input_id.size() # (b,t)

        assert in_t <= self.block_size,"cannot forward, model block size is exhuasted"

        #encoder
        in_x = self.input_embedding(in_id) #(b,t c)==(b,t,n_embd)
        in_position =self.input_positional_embedding[:,:t_in,:] #each position maps to a learnable vector
        in_x = self.input_ln_embd(in_x+in_position)
        in_x = self.input_drop(in_x)
        encoded_input = self.encoder_block(in_x) #serve as


        #decoder
        out_b, out_t =out_id.size()
        assert out_t <= self.block_size, "cannot forward, model block size is exhuasted"

        out_x = self.output_embedding(out_id)
        out_position = self.output_positional_embedding[:,:out_t,:]
        out_x = self.output_ln_embd(out_x+out_position)
        out_x = self.output_drop(out_x)
        out_x = self.decoder_block(out_x,encoded_input)  #(B,T,C)

        logits = self.decoder_head(out_x) # (b, t, n_embd) * (n_embd,n_vocab) = (b,t, n_vocab)

        loss = 0
        #if we are given some desired targets also calculate the loss
        if target is not None:
            loss = F.cross_entropy(logits.view(-1，logits.size(-1)), #(b*t, n_vocab)
                                  targets.view(-1),ignore_index=0)   #(b*t)

        return logits,loss
    """