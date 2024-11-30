import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

current_directory = os.getcwd()
root_directory = os.path.dirname(current_directory)

class ImageDataset(Dataset):
    def __init__(self, num_to_learn, mode,path_data,inverse=False):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []
        
        #path_blurry = os.path.join(root_directory, "SimulatedData", "blurred.npy")
        #path_original = os.path.join(root_directory, "SimulatedData", "original.npy")
        #path_data = os.path.join(root_directory, "SimulatedData", "rand_PSF.npy")
        
        #if not os.path.exists(path_blurry) or not os.path.exists(path_original):
        #    raise FileNotFoundError("Blurry or Original data file not found.")
        if not os.path.exists(path_data):
            raise FileNotFoundError("Blurry or Original data file not found.")
        
        #blurry_datas = np.load(path_blurry).astype(np.float32)
        #original_datas = np.load(path_original).astype(np.float32)
        datas = np.load(path_data,allow_pickle=True)#.astype(np.object)
        blurry_datas = np.stack(datas[:,1])
        original_datas = np.stack(datas[:,0])

        if inverse == False:
            idx_beg = 0;
            idx_end = num_to_learn;
        else:
            idx_beg = blurry_datas.shape[0]-num_to_learn;
            idx_end = blurry_datas.shape[0];


        for i in range(idx_beg,idx_end):
            blurry_data = blurry_datas[i]
            original_data = original_datas[i]
            
            img_blurry = (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min())
            img_original = (original_data - original_data.min()) / (original_data.max() - original_data.min())
            #img_blurry = blurry_data/blurry_data.max()
            #img_original = original_data/original_data.max()
            
            img_blurry = Image.fromarray(img_blurry)
            img_original = Image.fromarray(img_original)

            img_blurry = self.transform(img_blurry)
            img_original = self.transform(img_original)
            
            self.data.append((img_blurry, img_original))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
