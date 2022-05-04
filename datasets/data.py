from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import torch
class FaceData(Dataset):
    """
    parses csv file and return pytorch dataset
    """
    def __init__(self,path2file):
        """Initialize the BaseModel class.
        Parameters:
            path2file: path to data.
        """
        super(Dataset,self).__init__()
        data = pd.read_csv(path2file)
        data = data.dropna()
        data['Image'] = data['Image'].apply(lambda im :np.fromstring(im,sep=' '))#parse string to image
        images = data['Image'].apply(lambda img: np.reshape(img,(96,96)))
        self.images = torch.tensor(np.stack(images,axis=0),dtype=torch.double)
        self.images = torch.unsqueeze(self.images,dim=1)
        self.labels = torch.tensor(data.drop(columns=['Image']).to_numpy(),dtype=torch.double)
    def __getitem__(self, index):
        return self.images[index].double(),self.labels[index].double()
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data = FaceData('./data/training.csv')
    print(len(data))
   
