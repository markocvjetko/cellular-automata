import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image 
import torchvision.transforms as transforms

class MandelbrotDataset(Dataset):
    """Mandelbrot dataset, consists of a sequence of images of the Mandelbrot set.
    """
    
    def __init__(self, root: str):
        self.root = root
        #save paths to images in a list. Images are in the root directory and named zoom_001.png, zoom_002.png, etc.
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        #sort the list of paths so that the images are numerically ordered
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths) - 1
    
    def __getitem__(self, idx):
        '''input is the image at index idx
        label is the image at index idx + 1
        '''
        input = Image.open(self.img_paths[idx])
        label = Image.open(self.img_paths[idx + 1])

        #convert images to tensor
        input = transforms.ToTensor()(input)
        label = transforms.ToTensor()(label)

        #greyscale
        input = input[0]
        label = label[0]
        
        #add channel dimension
        input = input.unsqueeze(0)
        label = label.unsqueeze(0)
        
        return input, label
    
