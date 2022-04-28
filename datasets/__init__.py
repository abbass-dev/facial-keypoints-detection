from sklearn.utils import shuffle
from datasets.data import FaceData
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset

def create_dataset(**params):
    dataset_path = params['dataset_path']
    dataset = FaceData(dataset_path)
    if params['val_ratio'] is not False:
        spliter = ShuffleSplit(n_splits=1,test_size=0.2)
        for train_idx,val_idx in spliter.split(range(len(dataset))):
            train_ds = Subset(dataset,indices=train_idx)
            val_ds = Subset(dataset,indices=val_idx)
        train_dl = DataLoader(train_ds,**params['loader_params'])
        val_dl = DataLoader(val_ds,**params['loader_params'])
        return train_dl,val_dl
    else:
        return DataLoader(dataset,**params['loader_params'])
    
