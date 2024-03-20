import torch
import torchvision
from torchvision import transforms

from config import cfg

def get_imagenette():
    '''
        get dataloader
    '''
    print('==> Preparing ImageNette data..')
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


    trainset = torchvision.datasets.Imagenette(root='./data/pytorch',download=False,transform=transform, split="train")
    testset = torchvision.datasets.Imagenette(root='./data/pytorch',download=False,transform=transform, split="val")
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg['data']['batch_size'], shuffle=cfg['data']['shuffle'], num_workers=cfg['data']['num_workers'])
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg['data']['test_batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    return train_loader, test_loader
