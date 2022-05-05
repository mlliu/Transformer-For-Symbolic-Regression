import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils
import csv

#argparse parser for command-line options

argp = argparse.ArgumentParser()
argp.add_argument('function',
                  help ="whether to train, finetune or evaluate a model",
                  choices =["train","evaluate"])
#argp.add_argument('variant',
#                  help = "which variant of the model to run('vanilla' or 'synthesizer')",
#                  choices = ["vanillia","synthesizer"])
#argp.add_argument('corpus_path',
#                 help = "path of corpus for training"
                 #help="if specified, path of the model to load before finetuing/evaluation",
#                 default = None)
#argp.add_argument('--reading_params_path',
#    help="If specified, path of the model to load before finetuning/evaluation",
#    default=None)
#argp.add_argument('--writing_params_path',
#    help="Path to save the model after pretraining/finetuning", default=None)
#argp.add_argument('--finetune_corpus_path',
#    help="Path of the corpus to finetune on", default=None)
#argp.add_argument('--eval_corpus_path',
#    help="Path of the corpus to evaluate on", default=None)
#argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

#set the device:
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

#set the num_workers
num_workers =4 if torch.cuda.is_available() else 0

#keep the block_size 128
#why is the pretraining corpus always required(even if we're not pretraining?)
#it's because we use it as hack to always have the same vocabulary
#(that is the same mapping from character ro integer, and we build the vocab 
#from the pretraining corpus.)

block_size =64 #T
n_embd = 8 #embding dimensions  = number of categories **0.25 
n_head = 2 # since n_embd is so small, we can keep only 2 head
n_layer = 4 # layer


#points_path ='../data/datapoint_2.tsv'
#expr_pth = '../data/expr_2.tsv'

# open corpus
#points = open(args.points_path).read()
#expr = open(args.expr_path).read()

expr_path = '../data/expr_2.tsv'
expr = utils.read_tsv(expr_path)

points_path ='../data/datapoint_2.tsv'
points = utils.read_tsv(points_path,quoting=csv.QUOTE_NONNUMERIC)#points is numerical values

train_dataset = dataset.NameDataset(points,expr,block_size)

mconf = model.TransformerConfig(train_dataset.vocab_size,train_dataset.block_size,
                                 n_layer=n_layer,n_head=n_head,n_embd=n_embd)
model = model.Transformer(mconf)

#pretrain
if args.function == 'train':
    tconf = trainer.TrainerConfig(max_epochs=1,batch_size=1,learning_rate=6e-3,lr_decay=True,warmup_tokens=512*20,final_tokens=200*len(train_dataset)*block_size,num_workers=num_workers)
    trainer = trainer.Trainer(model, train_dataset, None,tconf )
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)


elif args.function =='evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path,map_location=torch.device('cpu')))
    correct =0
    total =0
    with open(args.outputs_path,'w') as fout:
        predictions=[]
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x+'??'
            # stoi character to integer
            x = torch.tensor([train_dataset.stoi[s] for s in x],stype = torch.long)[None,...].to(device)
            pred = utils.samople(model,x, 32,sample=False)[0]
            # itos integer to character
            completion=''.join()([train_dataset.itos[int(i)] for i in pred])
            pred = completion.split('??')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
            
        total, correct = utils.evaluate_places(args.eval_corpus_path,predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))
        
            

    