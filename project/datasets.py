import os
import torch as th
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LicensePlateDetectionDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, target_size=224) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.target_size = target_size


    def __len__(self) -> int:
        return len(self.metadata)


    def __getitem__(self, index):
        if th.is_tensor(index):
            index = index.tolist()
            
        img_id = self.metadata.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path)
        
        width, height = img.size
       
        x_scale = self.target_size / width
        y_scale = self.target_size / height

        ymin, xmin, ymax, xmax = self.metadata.iloc[index, 1:].values.astype(int)

        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        
        bbox = th.tensor([xmin, ymin, xmax, ymax]).float()
        
        if self.transform:
            img = self.transform(img)

        return img, bbox
