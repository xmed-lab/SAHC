
from enum import Flag
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection  import KFold

from dataset import *
from prototype import hierarch_train,base_predict
from utils import *
from hierarch_tcn2 import Hierarch_TCN2
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125
# print(device)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='hierarch_train')
parser.add_argument('--dataset', default="m2cai16")
parser.add_argument('--dataset_path', default="./datasets/{}/")

# parser.add_argument('--dataset', default="cholec80")
# parser.add_argument('--dataset_path', default="./datasets/cholec80")
parser.add_argument('--sample_rate', default=5, type=int)
parser.add_argument('--test_sample_rate', default=5, type=int)
parser.add_argument('--refine_model', default='gru')
parser.add_argument('--num_classes', default=7)
parser.add_argument('--model', default="Hierarch_TCN2")
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--epochs', default=100)
parser.add_argument('--gpu', default="3", type=str)
parser.add_argument('--combine_loss', default=False, type=bool)
parser.add_argument('--ms_loss', default=True, type=bool)

parser.add_argument('--fpn', default=True, type=bool)
parser.add_argument('--output', default=False, type=bool)
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--trans', default=False, type=bool)
parser.add_argument('--prototype', default=False, type=bool)
parser.add_argument('--last', default=False, type=bool)
parser.add_argument('--first', default=False, type=bool)
parser.add_argument('--hier', default=False, type=bool)
####ms-tcn2
parser.add_argument('--num_layers_PG', default="11", type=int)
parser.add_argument('--num_layers_R', default="10", type=int)
parser.add_argument('--num_R', default="3", type=int)

##Transformer
parser.add_argument('--head_num', default=8)
parser.add_argument('--embed_num', default=512)
parser.add_argument('--block_num', default=1)
parser.add_argument('--positional_encoding_type', default="learned", type=str, help="fixed or learned")
args = parser.parse_args()

learning_rate = 5e-5
epochs = 100
refine_epochs = 40

f_path = os.path.abspath('..')
root_path = f_path.split('surgical_code')[0]

if args.dataset == 'm2cai16':
    refine_epochs = 15 # early stopping
    args.num_classes = 8
   
    
loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')


num_stages = 3  # refinement stages
if args.dataset == 'm2cai16':
    num_stages = 2 # for over-fitting
num_layers = 12 # layers of prediction tcn e
num_f_maps = 64
dim = 2048
sample_rate = args.sample_rate
test_sample_rate = args.test_sample_rate
num_classes = len(phase2label_dicts[args.dataset])
args.num_classes = num_classes
# print(args.num_classes)
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R



print(args)

base_model=Hierarch_TCN2(args,num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)



if args.action == 'hierarch_train':
 
    video_traindataset = TestVideoDataset(args.dataset, args.dataset_path.format(args.dataset) + '/train_dataset', sample_rate, 'video_feature')
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=False, drop_last=False)
    video_testdataset = TestVideoDataset(args.dataset, args.dataset_path.format(args.dataset) + '/test_dataset', test_sample_rate, 'video_feature')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False) 
    model_save_dir = 'models/{}/'.format(args.dataset)
    hierarch_train(args, base_model, video_train_dataloader, video_test_dataloader, device, save_dir=model_save_dir, debug=True)

elif args.action == 'hierarch_predict':
   
    # print('ssss')
   
    model_path = '' # use your model
   
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =TestVideoDataset(args.dataset, root_path+ args.dataset_path.format(args.dataset) + '/test_dataset', test_sample_rate, 'video_feature')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_predict(base_model,args, device, video_test_dataloader)





        

    
    
