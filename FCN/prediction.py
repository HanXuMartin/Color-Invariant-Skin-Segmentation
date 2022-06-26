import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from tqdm import tqdm

# before 06 after 02

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints/fcn_model_2.pt')  #Load the model
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ =='__main__':
    img_file = 'C:/Users/xuhan/Desktop/ColorSpace_github/test_images' #your testing file

    img_names = os.listdir(img_file) 
        
    save_file = './output'
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    for img_name in tqdm(img_names):
        imgA = cv2.imread(os.path.join(img_file,img_name))
        size = imgA.shape

        imgA = cv2.resize(imgA, (160, 160))
        
        imgA = transform(imgA)
        imgA = imgA.to(device)
        imgA = imgA.unsqueeze(0)
        output = model(imgA)
        output = torch.sigmoid(output)
        output_np = output.cpu().detach().numpy().copy() 
        output_np = np.squeeze(output_np)*255
       

        output_npA = output_np[0]
        output_npB = output_np[1]

        
        output_sigmoid = output_npA/(output_npA+output_npB)*255
        
        cv2.imwrite(os.path.join(save_file,img_name), output_sigmoid)
        
        
        output = cv2.imread(os.path.join(save_file,img_name),0)
        output = cv2.resize(output,(size[1],size[0]))
        ret,output = cv2.threshold(output,127,255,cv2.THRESH_BINARY)
            
        cv2.imwrite(os.path.join(save_file,img_name), output)
    
        # break

