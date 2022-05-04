import torch
from datasets import create_dataset
from datasets import create_dataset
from utils import parse_configuration
from keyPoint import KeyPoint
'''
calculating model accuraccy 
Metric: key wise distance , considered accurate prediction 
if distance below some threshold 
'''
def validate():
    print('Initializing dataset...')
    #params = parse_configuration('./config.json')
    val_ds,train_ds = create_dataset(**params['train_dataset_params'])
    print('The number of training samples = {0}'.format(len(train_ds.dataset)))
    print('Initializing model...')
    model = torch.load('./key2.pt')
    model.device = 'cpu'
    running_metric = 0
    for image,label in val_ds:
        output = model(image)
        batch_size = output.shape[0]
        output = torch.reshape(output,(batch_size,2,-1)).permute(0,2,1)
        label = torch.reshape(label,(batch_size,2,-1)).permute(0,2,1)
        dis = torch.sqrt(torch.sum(torch.pow(output-label,2),dim=2))
        accu = dis < 2.5 #accurate if the distance less than 3 pixels
        running_metric += torch.sum(accu)
    print(f"validation accuract = {running_metric/(len(val_ds.dataset)*15)}")

