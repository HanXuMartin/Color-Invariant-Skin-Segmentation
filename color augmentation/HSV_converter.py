# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:27:28 2020

@author: xuhan
"""

import cv2
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def Change_HUE(image, rgb_path, hue_path):
    img = cv2.imread(os.path.join(rgb_path,image))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
    dhue = 30
    for i in range(int(180/dhue)):
        img_hsv[:,:,0] += dhue
        img_hsv[:,:,0] -= (img_hsv[:, :, 0] > 180) * 180

        img_hue = cv2.cvtColor(img_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)      
        save_hue = image.split('.')[0]+'_%d' %(dhue*(i+1))+'.png'
        
        if not os.path.exists(hue_path+'/'+'%d' %(dhue*(i+1))):
            os.makedirs(hue_path+'/'+'%d' %(dhue*(i+1)))
                        
        save_path = os.path.join(hue_path+'/'+'%d' %(dhue*(i+1)),save_hue)
        # print(save_path)
        cv2.imwrite(save_path,img_hue)

def Change_Saturation(image, rgb_path, saturation_path):
    img = cv2.imread(os.path.join(rgb_path,image))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
    image_H, image_S, image_V = cv2.split(img_hsv)
    dsat = 0.2
    for i in range(6):
        # print(dsat)
        change_s = image_S*(1-i*dsat)
        img_change = cv2.merge([image_H,change_s,image_V])
        img_sat = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        # save_saturation = image.split('.')[0]+'_%.1f' %(i*dsat)+'.png'
        if not os.path.exists(saturation_path+'/'+'%.1f' %(1-i*dsat)):
            os.makedirs(saturation_path+'/'+'%.1f' %(1-i*dsat))
            
        save_path = os.path.join(saturation_path+'/'+'%.1f' %(1-i*dsat),image)
        # print(save_path)
        cv2.imwrite(save_path,img_sat)



def Change_Value(image, rgb_path, value_path):
    img = cv2.imread(os.path.join(rgb_path,image))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
    image_H, image_S, image_V = cv2.split(img_hsv)
    dvalue = 0.2
    for i in range(int(1/dvalue)):
        # print(dsat)
        change_v = image_V*(1-i*dvalue)
        img_change = cv2.merge([image_H,image_S,change_v])
        img_value = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        # save_value = image.split('.')[0]+'_%.1f' %(i*dvalue)+'.png'
        if not os.path.exists(value_path+'/'+'%.1f' %(1-i*dvalue)):
            os.makedirs(value_path+'/'+'%.1f' %(1-i*dvalue))
            
        save_path = os.path.join(value_path+'/'+'%.1f' %(1-i*dvalue),image)
        # print(save_path)
        cv2.imwrite(save_path,img_value)



if __name__ == "__main__":
    image_path = 'G:/UNet2/data/ECU/test/image'
    save_path = 'G:/UNet2/data/ECU/test/HSV2'
    # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
    # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
    image_names = os.listdir(image_path)
    dH = 30
    dS = 0.2
    dV = 0.2
    # print(image_names)
    for image in tqdm(image_names):
        # Change_Value(image, image_path, value_path)
        # Change_Saturation(image, image_path, sat_path)
        # break
        # print()
        img = cv2.imread(os.path.join(image_path,image))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
        image_H, image_S, image_V = cv2.split(img_hsv)
        for h in range(6):
            change_H = image_H + dH*h
            change_H -= (change_H > 180) * 180

            for s in range(6):
                change_S = image_S*(1-s*dS)
                for v in range(5):
                    change_V = image_V*(1-v*dV)
                    img_change = cv2.merge([change_H,change_S,change_V])
                    img_change = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
                    change_path = save_path+'/'+'H%dS%.1fV%.1f' %(dH*h, 1-dS*s, 1-dV*v)
                    if not os.path.exists(change_path):
                        os.makedirs(change_path)
                    

                    cv2.imwrite(change_path+'/'+image,img_change)

        # break
    # 
    
    
    
    
    
    
    
    
    
    
    
    
    

