import torch
from datasets import create_dataset
from utils import parse_configuration
from model import FaceNet
from torch.optim import Adam
import torch.nn.functional as F
class Trainer():
    def __init__(self,model,train_ds,val_ds=None,**params):
        self.number_of_epoch = params['number_of_epoch']
        self.lr = params['lr']

        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.len_data = len(self.train_ds.dataset)
        self.optm = Adam(self.model.parameters(),self.lr)
        self.loss_history = []

    def loss_func(self,pred,target):
        return F.mse_loss(pred,target,reduction='sum')

    def epoch_loss(self):
        for image,target in self.train_ds:
            pred = self.model(image)
            running_loss = 0
            loss = self.loss_func(pred,target)

            if self.optm is not None:
                self.optm.zero_grad()
                loss.backward()
                self.optm.step()
            running_loss+=loss.item()
        return running_loss/self.len_data

    def train(self):
        print('start training')
        for i in range(self.number_of_epoch):
            print(f"epoch number = {i}")
            loss = self.epoch_loss()
            self.loss_history.append(loss)
            print(f"loss = {loss}")
        torch.save(model.state_dict(), './model.pt')
        return self.loss_history

    

        
if __name__=='__main__':
    params = parse_configuration('./config.json')
    train_ds,val_ds = create_dataset(**params['train_dataset_params'])
    model = FaceNet(15).double()
    trainer = Trainer(model,train_ds,val_ds,**params['train_params'])
    loss_history = trainer.train()
    print(loss_history)

