import random
import numpy as np
import torch
from torch.utils.data import Dataset
#import argaparse
import csv

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇bn t,□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

"""

# NameDataset should take the 
#class NameDataset(Dataset)


#mao-style Dataset: __getitem__() and __len__()
class NameDataset(Dataset):
    #the input is a text = open(args.pretrain_corpus_path).read()
    def __init__(self,points,expr,block_size):
        self.points = points
        self.expr = expr
        print(len(points))
        print(len(expr))
        #assert len(points) == len(expr)
       
        self.MASK_CHAR = u"\u2047" #doublequestionmark character for mask
        self.PAD_CHAR = u"\u25A1" #the empty square character, for pad
        
        #get all the characters
        chars = list(sorted(list(set([symbol  for each in self.expr for symbol in each ]))))
        #chars=list(sorted(list(set(expr))))
        """
        for each in data:
            expr = each.split('\t')
            chars.append(expr.split(' '))
            set(chars)
            
        chars =list(sorted(chars))
        """
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0,self.MASK_CHAR)
        chars.insert(0,self.PAD_CHAR)
        
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}
        
        data_size, vocab_size = len(expr),len(chars)
        
        print('data has %d characters, %d unique',(data_size,vocab_size))
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        
    def __len__(self):
        #return the length of the dataset
        return len(self.expr)
    # when accessed with dataset[idx], could read the idx-th image and its corresponding lable
    #from a folder on the disk
    def __getitem__(self,idx):
        #inp,oup = self.data[idx].split('\t')
        
        x1 = self.points[idx]# + self.PAD_CHAR*(self.bloack_size-len(inp))
        #assert(len(x1) == self.block_size) # assume that the length of x1 (datapoint) equals to block_size
        #                     *n_embd              # otherwise, we may need to pad it use 0
        
        x2 = [self.PAD_CHAR] + self.expr[idx] #caution: teach force
        x2 = x2 + [self.PAD_CHAR]*(self.block_size - len(x2)) 
        y = x2[1:]+ [self.PAD_CHAR]
        #print(x2)
        #print(y)
        #print(x1)
        #x1 = torch.tensor(x1,dtype = torch.float32)
        x1 = torch.FloatTensor(x1)
        x2 = torch.tensor([self.stoi[c] for c in x2],dtype = torch.long)
        y  =  torch.tensor([self.stoi[c] for c in y],dtype = torch.long)
        return x1, x2, y
        
    """
    def __getitem__(self,idx):
        inp,oup = self.data[idx].split('\t')
        x = inp +self.MASK_CHAR +oup +self.MASK_CHAR
        x = x +self.PAD_CHAR *(self.block_size-len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):] #caution: teach force
        
        x=x[:-1]
        
        x = torch.tensor([self.stoi[c] for c in x],dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y],stype = torch.long)#float32
        return x,y
     """
if __name__ == "__main__":
    points_path ='../data/datapoint_2.tsv'
    expr_path = '../data/expr_2.tsv'

    # open corpus this way didnot work, use csv methods,
    #points = open(points_path).read()
    #expr = open(expr_path).read()
    #expr = expr.split('\t')
    
    #read data 
    expr = []
    with open(expr_path) as expr_file:
        expr_data = csv.reader(expr_file, delimiter="\t")
     
        # printing data line by line
        for line in expr_data:
            expr.append(line)
    points =[]
    with open(points_path) as p_file:
        p_data = csv.reader(p_file, delimiter="\t",quoting=csv.QUOTE_NONNUMERIC)# to read numerical data
     
        # printing data line by line
        for line in p_data:
            points.append(line)
    #print(expr[0])
   
    
    
    
    block_size =128
    dataset = NameDataset(points,expr,block_size)
    idx =0
    x1,x2,y = dataset.__getitem__(idx)
    print(x1)
    print(x2)
    print(y)
    