import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImgDirDataset(Dataset):
     def __init__(self, path_to_dir):
          self.path_to_dir = path_to_dir
          self.ids = []
          
          # image_ext = '.jpg'
          for file in os.listdir(path_to_dir):
               f_name, ext = os.path.splitext(file)
               if f_name not in self.ids:
                    self.ids.append(f_name)   
                    
          self.img_transform = transforms.Compose([
               transforms.v2.RGB(),
               transforms.PILToTensor(),
               transforms.ConvertImageDtype(torch.float32)
          ])
                    
     def __len__(self):
          return len(self.ids)
     
     def __getitem__(self, idx):
          id = self.ids[idx]
          image_path = f"{self.path_to_dir}/{id}.jpg"
          text_path = f"{self.path_to_dir}/{id}.txt"
          
          image = Image.open(image_path)
          t_image = self.img_transform(image)
          
          with open(text_path, 'r') as text_file:
               text = text_file.read().strip()
               
          return t_image, text
               
          
