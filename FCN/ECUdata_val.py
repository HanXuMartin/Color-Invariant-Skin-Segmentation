import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
from onehot import onehot
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class Dataset(Dataset):

    def __init__(self, transform=None):
       self.transform = transform 
       self.train_file = 'G:/UNet2/data/ECU/valid/colorjitter'
       self.mask_file = 'G:/UNet2/data/ECU/valid/colorjitter_label'
    def __len__(self):
       return len(os.listdir(self.train_file))

    def __getitem__(self, idx):
        img_name = os.listdir(self.train_file)
        img_name = img_name[idx]
        imgA = cv2.imread(os.path.join(self.train_file, img_name))
        # print(img_name)
        # plt.imshow(imgA)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread(self.mask_file+'/'+img_name[0:7]+'_gt.png', 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)    
        item = {'A':imgA, 'B':imgB}
        return item

training_data = Dataset(transform)
dataloader_val = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=4)
if __name__ =='__main__':
    for batch in dataloader:
        print(len(dataloader))
        break






