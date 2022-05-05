import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    """
    A vanillia multi-head masked attention layer with a project at the end
    causal: an optional mask
    encoder: casual=true 
    decoder: casual=false -- a triangular inferior attention
    In facebook's code, they seems consider use length as mask, later discuss
    """
    def __init__(self,config):
        super().__init__()
        #key query value projections for all head
        self.key = nn.Linear(config.n_embd,config.n_embd)
        self.query = nn.Linear(config.n_embd,config.n_embd)
        self.value = nn.Linear(config.n_embd,config.n_embd)
        
        #regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        #register_buffer(name, tensor, persistent=True)
        #causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size,config.block_size))
                                        .view(1,1,config.block_size,config.block_size))
        
        #output projection why we need this?
        self.proj = nn.Linear(config.n_embd,config.n_embd)
        
        self.n_head = config.n_head
    
    #if casual is True, assert encoded_input != None, we're in a decoder block 
    def forward(self,x,causal=None,encoded_input=None):
        B,T,C = x.size() #C is n_embd
        
        #calculate K Q V for all heads in batches and 
        #use torch.transpose(input, dim0, dim1) to swap dim0 and dim1
        Q = self.query(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)#(B,nh,T,hs)
        if encoded_input is None:
            K = self.key(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
            V = self.value(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        else:
            assert(encoded_input.size() == B,T,C)
            K = self.key(encoded_input).view(B,T,config.n_head,C//self.n_head).transpose(1,2)
            V = self.value(encoded_input).view(B,T,config.n_head,C//self.n_head).transpose(1,2)
        #matmul(dot product) and scale
        att = (Q @ K.transpose(-2,-1))* (1.0/ math.sqrt(K.size(-1)))
        #mask (optional)
        #Tensor.masked_fill(mask, value)
        if causal:
            att.masked_fill(self.mask[:,:,:T,:T]==0, -float('inf'))#-1e10)#todo: just use float('-inf') instead?
        #softmax #matmul with Q
        att = F.softmax(att,dim=-1)
        att = self.attn_drop(att)
        y = att @ V #(B,nh,T,T) x (B,nh,T,hs)  = (B,nh,T,hs)
        #re-assemble all head outputs side by side
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        #output projection
        y = self.resid_drop(self.proj(y))
        
        return y
        
            