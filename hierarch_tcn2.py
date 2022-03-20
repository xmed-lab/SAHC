import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torchvision
from decoder import TransformerDecoder, TransformerDecoderLayer
from PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
import copy
import numpy as np
class FPN(nn.Module):
    def __init__(self,num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,W = y.size()
        return F.upsample(x, size=W, mode='linear') + y

    def forward(self,out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1,p2,p3,p4]

class Hierarch_TCN2(nn.Module):

    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(Hierarch_TCN2, self).__init__()
        # self.PG = Prediction_Generation(args, num_layers_PG, num_f_maps, dim, num_classes)
        self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)

        self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.first_linear = nn.Linear(num_f_maps*4, num_f_maps, 1)
        self.conv_out1 = nn.Conv1d(num_f_maps*3, num_classes, 1)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(args, num_layers_R, num_f_maps, num_classes, num_classes,self.conv_out)) for s in range(num_R)])
        self.use_fpn = args.fpn
        self.use_output = args.output
        self.use_feature = args.feature
        self.use_trans = args.trans
        # self.prototpye=[]
        if args.fpn:
            self.fpn = FPN(num_f_maps)
        if args.trans:
            self.query = nn.Embedding(num_classes, num_f_maps)
        

            if args.positional_encoding_type == "learned":
                self.position_encoding = LearnedPositionalEncoding(
                    19971, num_f_maps
                )
            elif args.positional_encoding_type == "fixed":
                self.position_encoding = FixedPositionalEncoding(
                num_f_maps,
                )
            else:
                self.position_encoding=None
            print('position encoding :',  args.positional_encoding_type)
            decoder_layer = TransformerDecoderLayer(num_f_maps, args.head_num, args.embed_num,
                                            0.1, 'relu',normalize_before=True)
            decoder_norm = nn.LayerNorm(num_f_maps)
            self.decoder = TransformerDecoder(decoder_layer, args.block_num, decoder_norm,
                                    return_intermediate=False)
        self.prototpye = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=True)

    def forward(self, x):
        out_list = []
        f_list = []
        x = x.permute(0,2,1)

        f, out1 = self.PG(x)
    
       
        f_list.append(f)
        if not self.use_fpn:
            out_list.append(out1)
      
        # print(out.size())
        
       
        for R in self.Rs:
            # F.softmax(out, dim=1)
            if self.use_output:
                f, out1 = R(out1)
                out_list.append(out1)
                # print(out1.size())
            else:
                f, out1 = R(f)
            # print(f.size())
            # print(out.size())
           
            f_list.append(f)
            if not self.use_fpn:
                out_list.append(out1)
            # outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        # print(len(out_list))
        if self.use_fpn:
            f_list = self.fpn(f_list)
            for f in f_list:
                # print(f.size())
                out_list.append(self.conv_out(f))
        # sss
        if self.use_feature:
            last_feature = f_list[-1]
            refine_out = torch.matmul(self.prototpye.transpose(1,2),last_feature)
            out_list[-1] = 0.5*out_list[-1] + 0.5*refine_out

        # print(len(f_list))


        if self.use_trans:
           
           
            for i in range(len(f_list)):
                if self.position_encoding == None:
                    f_list[i] =  f_list[i]
                else:
                    # print(f_list[i].size())
                    f_list[i] = self.position_encoding(f_list[i])
            # query_embed = self.query.weight.unsqueeze(1).repeat( 1, batch_size, 1)
            
            # first_feature = f_list[0]
            first_feature_list= []
            first_feature_list.append(f_list[0])
            first_feature = f_list[0].permute(2,0,1)
            # print(len(f_list))
            # sss
            for i in range(1, len(f_list)):
                middle_feature = f_list[i]

                first_feature = self.decoder(first_feature, middle_feature, 
                    memory_key_padding_mask=None, pos=None, query_pos=None)
                # print(first_feature.size(),middle_feature.size())
                
                # attention_w = torch.matmul(first_feature.transpose(1,2), middle_feature)
                # attention_w = F.softmax(attention_w,dim=2)
                # new_first_feature = torch.matmul(attention_w, middle_feature.transpose(1,2))
                # print(new_first_feature.transpose().size())
                # ssss
                # first_feature_list.append(new_first_feature.transpose(1,2))
                # first_feature_list.append(new_first_feature.permute(1,2,0))
                # last_feature = f_list[-1]
                # middle_feature = f_list[-2]
                # # print(pos_embd.size())
                
                # # x = self.conv_out(out) # (bs, c, l)
                # # out = last_feature.permute(2,0,1)
                # first_feature = f_list[0].permute(2,0,1)
                # # print(first_feature.size(), last_feature.size())
                # first_feature = self.decoder(first_feature, last_feature, 
                #     memory_key_padding_mask=None, pos=None, query_pos=None)
                # f_list[0] = first_feature.permute(1,2,0)
        
            # f_list[0] = torch.cat(first_feature_list,dim=1)
            # f_list[0] = torch.stack(first_feature_list,dim=1).sum(dim=1)
            
            # print(f_list[0].size())
            # print(f_list[1].size())
            # reduced_first_feature = self.first_linear(f_list[0].transpose(1,2)).transpose(1,2)
            # reduced_first_feature=f_list[0]
            reduced_first_feature=first_feature.permute(1,2,0)
            out_list[0] = self.conv_out(reduced_first_feature)
            # for idx, f in enumerate(f_list):
            #     if idx == 0:
            #         out_list.append(self.conv_out1(f))
            #     else:
            #         out_list.append(self.conv_out(f))

                
            # out_list[-1] = pro
        return out_list, f_list, self.prototpye

class BaseCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        print(num_layers)
        super(BaseCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        # self.downsample = nn.Linear(num_f_maps,num_f_maps, kernel_size=3, stride=2,dilation=3)
        # self.center = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=False)
        self.num_classes = num_classes
        

    def forward(self, x, labels=None, mask=None,test=False):
        # x = x.permute(0,2,1) # (bs,l,c) -> (bs, c, l)
        
        if mask is not None:
            # print(x.size(),mask.size())
            x = x * mask
        
        x= x.unsqueeze(3) # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)

      
        x = self.conv_out(out) # (bs, c, l)

     
        return out, x  


class Prediction_Generation(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()
        
        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            [copy.deepcopy(DilatedResidualCausalLayer(2**(num_layers-1-i), num_f_maps, num_f_maps))
            for i in range(num_layers)]
        ))

        # self.conv_dilated_1 = nn.ModuleList((
        #     nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
        #     for i in range(num_layers)
        # ))
        self.conv_dilated_2 = nn.ModuleList((
            [copy.deepcopy(DilatedResidualCausalLayer(2**i, num_f_maps, num_f_maps))
            for i in range(num_layers)]
        ))
        # self.conv_dilated_2 = nn.ModuleList((
        #     nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
        #     for i in range(num_layers)
        # ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        
        f = self.conv_1x1_in(x)
        
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        
        out = self.conv_out(f)

        return f, out

class Refinement(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes, conv_out):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualCausalLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.conv_out = conv_out
        self.max_pool_1x1 = nn.AvgPool1d(kernel_size=7,stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)
        
        return f, out
    
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return x + out

class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, padding=None):
        super(DilatedResidualCausalLayer, self).__init__()
        if padding == None:
            
            self.padding = 2 * dilation
        else:
            self.padding=padding
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation) #
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0) # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)