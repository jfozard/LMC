
import torch

import segmentation_models_pytorch as smp

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import pickle
import random
import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys

S=512


def stack_im(im):
    o = np.concatenate([np.concatenate(im[i:i+2], axis=1) for i in range(0, len(im), 2)], axis=0)
    return o

out_map = {'Nucleus':0, 'Mitochondria':1, 'Tubulin':2, 'Actin':3}

def percentile_normalization(image, pmin=0.01, pmax=99.9, axis=None, dtype=np.uint16 ):
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
        image = TF.normalize(image, image.mean(dim=(1,2)), torch.clip(image.std(dim=(1,2)), min=1e-6)) # 0.485 0.229
        out = TF.crop(out, i, j, h, w)
#        out = TF.normalize(out, out.mean(dim=(1,2)), out.std(dim=(1,2))) # 0.485 0.229

        out_class = torch.tensor([out_map[out_im[1]]])
        return image, out, out_class, torch.tensor([idx])

model = smp.Unet(
#    encoder_name="efficientnet-b7", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_name=sys.argv[1], #"tu-tf_efficientnetv2_xl", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#    encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4, # model output channels (number of classes in your dataset)
    activation='sigmoid'
)



import torch
import torch.nn as nn
from torch.optim import Adam

def replace_batchnorm(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module)
        
        if isinstance(module, torch.nn.BatchNorm2d):
            model._modules[name] = torch.nn.GroupNorm(8, module.num_features)
#            model._modules[name] = torch.nn.LayerNorm(32, module.num_features)

    return model 
    
#model = replace_batchnorm(model)

full_dataset = NPZDataset('data/')

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, drop_last=True)

# Loss and optimizer
criterion = nn.L1Loss()
criterion2 = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=1e-4)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

state = torch.load(sys.argv[2])
print(list(state['model_state_dict']))
model.load_state_dict(state['model_state_dict'])

print(model)

model.eval()
with torch.no_grad():
    running_test_loss = 0.0
    for images, masks, out_class_i, idx in tqdm.tqdm(test_loader):
        images, masks, out_class_i = images.to(device), masks.to(device), out_class_i.to(device)

        # Forward pass
        outputs = model(images)
        out_class =  out_class_i.unsqueeze(2).unsqueeze(3).expand(-1, -1, outputs.shape[2], outputs.shape[3])
        outputs = torch.gather(outputs, 1, out_class)

        loss = criterion(outputs, masks)

        x = outputs
        y = masks

        vx = x - torch.mean(x, dim=(2,3), keepdim=True)
        vy = y - torch.mean(y, dim=(2,3), keepdim=True)

        cost  = torch.sum(vx * vy, dim=(2,3)) / (torch.norm(vx, dim=(2,3))* torch.norm(vy, dim=(2,3)) + 1e-6) 

        loss -= cost.mean()

        running_test_loss += loss.item() * images.size(0)
        print('test loss', running_test_loss)


fig, ax = plt.subplots(1,3, figsize=(12,4))
# Training loop
num_epochs = 1000
ax[0].cla()
ax[0].imshow(stack_im(images[:,0].detach().cpu().numpy()))
for i in range(idx.shape[0]):
    ax[0].text((i%2)*S+10, (i//2)*S+50, str(idx[i].item()), c='w')
ax[1].cla()
ax[1].imshow(stack_im(outputs[:,0].detach().cpu().numpy()))
ax[2].cla()
ax[2].imshow(stack_im(masks[:,0].detach().cpu().numpy()))
for i in range(out_class_i.shape[0]):
    ax[2].text((i%2)*S+10, (i//2)*S+50, str(out_class_i[i].item()), c='w')
plt.draw()
plt.show()

