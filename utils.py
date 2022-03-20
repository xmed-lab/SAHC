from matplotlib import pyplot as plt
from matplotlib import *
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from MulticoreTSNE import MulticoreTSNE as TSNE

import seaborn as sns
phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
}
def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] for label in labels]
    return phases
max_pool = nn.MaxPool1d(kernel_size=13,stride=5,dilation=3)


path_p= "/home/xmli/phwang/ntfs/xinpeng/code/casual_tcn/results/m2cai16/eva/resize/"
def fusion(predicted_list,labels,args):

    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0/len_layer for i in range (0, len_layer)]
    # print(weight_list)
    num=0
    for out, w in zip(predicted_list, weight_list):
        resize_out =F.interpolate(out,size=labels.size(0),mode='nearest')
        resize_out_list.append(resize_out)
        # align_corners=True
        # print(out.size())
        resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='linear',align_corners=False)
        if out.size(2)==labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            # resize_label = max_pool(labels_list[-1].float().unsqueeze(0).unsqueeze(0))
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='nearest')
            # resize_label2 = F.interpolate(resize_label,size=labels.size(0),mode='nearest')
            # ,align_corners=True
            # print(resize_label.size(), resize_label2.size())
            # print((resize_label2 == labels).sum()/labels.size(0))
            # with open(path_p+'{}.txt'.format(num),"w") as f:
            #     for labl1, lab2 in zip(resize_label2.squeeze(), labels.squeeze()):
            #         f.writelines(str(labl1)+'\t'+str(lab2)+'\n')
            # num+=1
            labels_list.append(resize_label.squeeze().long())
            # labels_list.append(labels.squeeze().long())
        # print(resize_label.size(), out.size())
        # labels_list.append(labels.squeeze().long())
        # assert resize_out.size(2) == resize_label.size(0)
        # assert resize_label.size(2) == out.size(2)
        # print(out.size())
        # print(resize_label.size())
        # print(resize_out.size())
        # all_out_list.append(out)
        # all_out_list.append(resize_out)

        all_out_list.append(out)
        # resize_out=out
        # all_out = all_out + w*resize_out
    
    # sss
    return all_out, all_out_list, labels_list

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    # dist = 1. - similiarity
    return similiarity




def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)
    
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    color_map = plt.cm.tab10

#     axprops = dict(xticks=[], yticks=[0,0.5,1], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=15)
    fig = plt.figure(figsize=(15, (num_pics+1) * 1.5))

    interval = 1 / (num_pics+2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1-i*interval, 0.8, interval - interval/num_pics]))
#         ax1.imshow([label], **barprops)
    titles = ['Ground Truth','Causal-TCN', 'Causal-TCN + PKI', 'Causal-TCN + MS-GRU']
    for i, label in enumerate(labels):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
#         axes[i].set_title(titles[i])
    
    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval/num_pics])
#     ax99.set_xlim(-len(confidence_score)/15, len(confidence_score) + len(confidence_score)/15)
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(-0.2, 1.2)
    ax99.set_yticks([0,0.5,1])
    ax99.set_xticks([])
 
     
    ax99.plot(range(len(confidence_score)), confidence_score)

    if save_path is not None:
        print(save_path)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
def PKI(confidence_seq, prediction_seq, transition_prior_matrix, alpha, beta, gamma): # fix the predictions that do not meet priors
    initital_phase = 0
    previous_phase = 0
    alpha_count = 0
    assert len(confidence_seq) == len(prediction_seq)
    refined_seq = []
    for i, prediction in enumerate(prediction_seq):
        if prediction == initital_phase:
            alpha_count = 0
            refined_seq.append(initital_phase)
        else:
            if prediction != previous_phase or confidence_seq[i] <= beta:
                alpha_count = 0
            
            if confidence_seq[i] >= beta:
                alpha_count += 1
            
            if transition_prior_matrix[initital_phase][prediction] == 1:
                refined_seq.append(prediction)
            else:
                refined_seq.append(initital_phase)
            
            if alpha_count >= alpha and transition_prior_matrix[initital_phase][prediction] == 1:
                initital_phase = prediction
                alpha_count = 0
                
            if alpha_count >= gamma:
                initital_phase = prediction
                alpha_count = 0
        previous_phase = prediction

    
    assert len(refined_seq) == len(prediction_seq)
    return refined_seq