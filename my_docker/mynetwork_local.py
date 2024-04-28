
#You can add your network here


import torch

import segmentation_models_pytorch as smp

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import pickle
import random
import math

import ttach as tta

S=512

out_map = {'Nucleus':0, 'Mitochondria':1, 'Tubulin':2, 'Actin':3}

def percentile_normalization(image, pmin=2, pmax=99.8, axis=None, dtype=np.uint16 ):
    '''
    Compute a percentile normalization for the given image.

    Parameters:
    - image (array): array of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed. 
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    '''

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_p  = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        print(f"Same min {low_p} and high {high_p}, image may be empty")

    else:
        dtype_max = np.iinfo(dtype).max
        img_norm = dtype_max * np.clip((image - low_p) / ( high_p - low_p ), 0, 1)
        img_norm = img_norm.astype(dtype)

    return img_norm

class NPZDataset(Dataset):
    def __init__(self, dataset_path):
        self.meta = pickle.load(open(dataset_path + 'meta.pkl', 'rb'))
        self.images = list(pickle.load(open(dataset_path + 'images.pkl', 'rb')).values())
        self.path = dataset_path

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        imdata = self.images[idx]
        in_im = random.choice(imdata['input_images'])

        with np.load(self.path + '/' + in_im[0] +'.npz') as in_npz:
            image = in_npz['arr_0'] / 65535 

        #print(imdata['output_images'])
        out_im = random.choice(imdata['output_images'])
        with np.load(self.path + '/' + out_im[0] +'.npz') as out_npz:
            out = out_npz['arr_0'] #/ 65535


        out = percentile_normalization(out) /65535
            
        # Convert to PyTorch tensors
        image = TF.to_tensor(image.astype(np.float32))
        out = TF.to_tensor(out.astype(np.float32))
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(S, S))
        #i = j = 1024
        image = TF.crop(image, i, j, h, w)
        image = TF.normalize(image, image.mean(dim=(1,2)), image.std(dim=(1,2))) # 0.485 0.229
        out = TF.crop(out, i, j, h, w)
#        out = TF.normalize(out, out.mean(dim=(1,2)), out.std(dim=(1,2))) # 0.485 0.229

        out_class = torch.tensor([out_map[out_im[1]]])
        return image, out, out_class, torch.tensor([idx])




def replace_batchnorm(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module)
        
        if isinstance(module, torch.nn.BatchNorm2d):
            model._modules[name] = torch.nn.GroupNorm(8, module.num_features)
#            model._modules[name] = torch.nn.LayerNorm(32, module.num_features)

    return model 
    


class MyNetWork():

    def __init__(self, load_path=None, load_path2=None):
        self.model = smp.Unet(
        # encoder_name="efficientnet-b7", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #encoder_name="efficientnet-b7", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="tu-maxvit_base_tf_512",
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4, # model output channels (number of classes in your dataset)
        activation='sigmoid'
        )

        self.model = replace_batchnorm(self.model)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)    
        self.model.load_state_dict(torch.load(load_path)['model_state_dict'])
        
        #self.model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean') 


        self.model2 = smp.Unet(
        encoder_name="efficientnet-b7", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1, # model output channels (number of classes in your dataset)
        activation='sigmoid'
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model2.to(self.device)    
        self.model2.load_state_dict(torch.load(load_path2)['model_state_dict'])

        
        
        
        self.model.eval()
        self.model2.eval()
        
    def forward(self, image, tl):
        with torch.no_grad():
            image = TF.to_tensor(image.astype(np.float32))
            o_shape = image.shape
            image = TF.normalize(image, image.mean(dim=(1,2)), image.std(dim=(1,2)))
            pad_h = int(math.ceil(image.shape[1]/32)*32 - image.shape[1])
            pad_w = int(math.ceil(image.shape[2]/32)*32 - image.shape[2])
            image = TF.pad(image, (0,0,pad_w,pad_h))
            print(image.shape)
            image = image.to(self.device)

            if True:
                out = torch.zeros((1,4, image.shape[1], image.shape[2]), device=self.device)
                ni = (image.shape[1]+511)//512
                nj = (image.shape[2]+511)//512
                for i in range(ni):
                    for j in range(nj):
                        end_i = min(image.shape[1], (i+1)*512)
                        end_j = min(image.shape[2], (j+1)*512)
                        start_i = end_i - 512
                        start_j = end_j - 512
                        tmp = image[:,start_i:end_i,start_j:end_j].unsqueeze(0).to(self.device)
                        print(tmp.shape)
                        out[:,:,start_i:end_i, start_j:end_j] = self.model(tmp)
            else:
                                   
                out = self.model(image.unsqueeze(0))
            out = torch.clip(out[:,:,:o_shape[1],:o_shape[2]], 0, 1)
            
            out2 = self.model2(image.unsqueeze(0))
            out2 = torch.clip(out2[:,:,:o_shape[1],:o_shape[2]], 0, 1)
        print(out.shape, out.min(), out.max())
        results = []
        #for i in range(out.shape[1]):
        results.append(out[0,0].cpu().numpy())
        results.append(out[0,1].cpu().numpy())
        results.append(out[0,2].cpu().numpy())
        results.append(out[0,3].cpu().numpy())
        return results
