from numpy.lib.function_base import append
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
import numpy as np
import torch
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




def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases



class TestVideoDataset(Dataset):
    def __init__(self, dataset, root, sample_rate, video_feature_folder):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []
        ###      
        self.video_names = []
        if dataset =='cholec80':
            self.hard_frame_index = 7
        if dataset == 'm2cai16':
            self.hard_frame_index = 8 

        video_feature_folder = os.path.join(root, video_feature_folder)
        label_folder = os.path.join(root, 'annotation_folder')
    
       
        num_len = 0
       
        ans = 0
        for v_f in os.listdir(video_feature_folder):
          
            
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
           
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            
         
            labels = self.read_labels(v_label_file_abs_path) 
            # 
            labels = labels[::sample_rate]
           
            videos = np.load(v_f_abs_path)[::sample_rate,]
           
            num_len += videos.shape[0]
           

            self.videos.append(videos)
           
            self.labels.append(labels)
            phase = 1
            for i in range(len(labels)-1):
                    if labels[i] == labels[i+1]:
                        continue
                    else:
                        phase += 1
          
            ans += 1
            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.videos)
       

  
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name
    
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels
