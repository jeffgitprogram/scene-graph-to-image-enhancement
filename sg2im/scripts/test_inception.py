import argparse
import os

import numpy as np
from skimage import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from IPython import display
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from torchvision.models import inception_v3

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir')
parser.add_argument('--img_size', default=299, type=int)
parser.add_argument('--batch_size', default=64, type=int)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_image(filename, loader, volatile=False):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor, volatile=volatile).unsqueeze(0)
    return image_var

class TestDataset(Dataset):
    def __init__(self, root_dir, loader):
        paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
        self.images = paths
        self.loader = loader
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = load_image(self.images[idx], self.loader).squeeze()
        return image
        
def inception_score(net, loader):
    up = torch.Upsample(size=(299,299), mode='bilinear')
    with torch.no_grad():
        scores = []
        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            images = up(images)
            score, _ = net(images)
            scores.append(score)
    scores = torch.cat(scores, 0)
    p_yx = F.softmax(scores)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score

def main(args):
    img_size = args.img_size
    loader = transforms.Compose([
#       transforms.Resize(img_size),
#       transforms.CenterCrop(img_size),
#       transforms.ToTensor(),
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    
    dataset = TestDataset(args.img_dir, loader)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    net = inception_v3(pretrained=True).to(device)
    net.eval()
    score = inception_score(net, dataloader)
    
    print('Inception score: %.4f' % score)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)