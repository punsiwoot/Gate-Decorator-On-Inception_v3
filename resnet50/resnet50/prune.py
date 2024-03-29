
# coding: utf-8

# In[ ]:


''' setting before run. every notebook should include this code. '''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import sys

_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('GGAP')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)

from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "0505_tock_more",
        "cuda": True,
        "seed": 0,
        "model_saving_interval": 90,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": True,
        "fp16": False
    },
    "model": {
        "name": "resnet50",
        "num_class": 1000,
        "pretrained": True
    },
    "train": {
        "trainer": "normal",
        "max_epoch": 90,
        "optim": "sgd",
        "steplr": {},
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "nesterov": False
    },
    "data": {
        "type": "imagenet",
        "shuffle": True,
        "batch_size": 64,
        "test_batch_size": 128,
        "num_workers": 8
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "sparse_lambda": 1e-3,
        "flops_eta": 0,
        "lr_min": 1e-3,
        "lr_max": 1e-2,
        "tock_epoch": 10,
        "T": 10,
        "p": 0.002
    }
})
from config import cfg


# In[ ]:


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from logger import logger
from main import set_seeds, recover_pack, adjust_learning_rate, _step_lr, _sgdr
from models import get_model
from utils import dotdict

from prune.universal import Meltable, GatedBatchNorm2d, Conv2dObserver, IterRecoverFramework, FinalLinearObserver
from prune.utils import analyse_model, DoRealPrune, finetune


# In[ ]:


set_seeds()
pack = recover_pack()


# In[ ]:


GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
    gbn.extract_from_bn()


# In[ ]:


pack.optimizer = optim.SGD(
    pack.net.parameters() ,
    lr=1,
    momentum=cfg.train.momentum,
    weight_decay=cfg.train.weight_decay,
    nesterov=cfg.train.nesterov
)


# ----

# In[ ]:


import uuid

def bottleneck_set_group(net):
    layers = [
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4
    ]
    for m in layers:
        masks = []
        for mm in m.modules():
            if mm.__class__.__name__ == 'Bottleneck':
                if mm.downsample is not None:
                    masks.append(mm.downsample._modules['1'])
                masks.append(mm.bn3)

        group_id = uuid.uuid1()
        for mk in masks:
            mk.set_groupid(group_id)

bottleneck_set_group(pack.net.module.backbone)


# In[ ]:


def clone_model(net):
    model = get_model()
    gbns = GatedBatchNorm2d.transform(model.module)
    model.load_state_dict(net.state_dict())
    return model, gbns


# In[ ]:


cloned, _ = clone_model(pack.net)
BASE_FLOPS, BASE_PARAM = analyse_model(cloned.module, input_size=(1, 3, 224, 224))
print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
print('%.3f M' % (BASE_PARAM / 1e6))
del cloned


# In[ ]:


def eval_prune(pack):
    cloned, _ = clone_model(pack.net)
    _ = Conv2dObserver.transform(cloned.module)
    cloned.module.backbone.fc = FinalLinearObserver(cloned.module.backbone.fc)
    cloned_pack = dotdict(pack.copy())
    cloned_pack.net = cloned
    Meltable.observe(cloned_pack, 0.001)
    Meltable.melt_all(cloned_pack.net)
    flops, params = analyse_model(cloned_pack.net.module, input_size=(1, 3, 224, 224))
    del cloned
    del cloned_pack
    
    return flops, params


# ----

# In[ ]:


import random
import torchvision
from torch.utils.data.sampler import Sampler
from torchvision import transforms

class SubsetSampler(Sampler):
    def __init__(self, dataset, batch_size, sample_per_class=50):
        self.dataset = dataset
        self.sample_per_class = sample_per_class
        self.indicator = None
        self.batch_size = batch_size
        self.make_subset()

    def make_subset(self):
        kits = [[] for i in range(len(self.dataset.class_to_idx))]

        for idx, (path, target) in enumerate(self.dataset.samples):
            kits[target].append(idx)

        r = []
        for k in kits:
            r.extend(random.sample(k, self.sample_per_class))

        for i in range(5):
            random.shuffle(r)

        self.indicator = r

    def __iter__(self):
        batch = []
        for idx in range(len(self.indicator)):
            batch.append(self.indicator[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
    
    def __len__(self):
        return len(self.indicator) // self.batch_size


def get_subset_loaders(root, sample_per_class=50):
    train_dir = os.path.join(root, 'train')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    sampler = SubsetSampler(train_dataset, cfg.data.batch_size, sample_per_class)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.data.num_workers,
        batch_sampler=sampler,
        pin_memory=True
    )

    return train_loader, sampler


# In[ ]:


tick_loader, Sampler = get_subset_loaders('./data/imagenet12', 200)
pack.tick_trainset = tick_loader

tock_loader, Sampler = get_subset_loaders('./data/imagenet12', 400)
pack.train_loader = tock_loader

prune_agent = IterRecoverFramework(pack, GBNs, sparse_lambda = cfg.gbn.sparse_lambda, flops_eta = cfg.gbn.flops_eta, minium_filter = 3)


# In[ ]:


LOGS = []
flops_save_points = set([70, 60, 50, 45, 42, 40, 38, 35, 32, 30])

iter_idx = 0
prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)
while True:
    left_filter = prune_agent.total_filters - prune_agent.pruned_filters
    num_to_prune = int(left_filter * cfg.gbn.p)
    info = prune_agent.prune(num_to_prune, tick=True, lr=cfg.gbn.lr_min * (cfg.data.batch_size // 64), test=False)
    flops, params = eval_prune(pack)
    info.update({
        'flops': '[%.2f%%] %.3f MFLOPS' % (flops/BASE_FLOPS * 100, flops / 1e6),
        'param': '[%.2f%%] %.3f M' % (params/BASE_PARAM * 100, params / 1e6)
    })
    LOGS.append(info)
    print('Iter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f' % 
          (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss']))
    
    iter_idx += 1
    if iter_idx % cfg.gbn.T == 0:
        print('Testing: %s' % str(pack.trainer.test(pack)))
        print('Tocking:')
        prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)
        tock_loader, Sampler = get_subset_loaders('./data/imagenet12', 400)
        pack.train_loader = tock_loader

    flops_ratio = flops/BASE_FLOPS * 100
    for point in [i for i in list(flops_save_points)]:
        if flops_ratio <= point:
            torch.save(pack.net.module.state_dict(), './logs/0505_tock_more/%s.ckp' % str(point))
            flops_save_points.remove(point)

    if len(flops_save_points) == 0:
        break
    
    tick_loader, Sampler = get_subset_loaders('./data/imagenet12', 200)
    pack.tick_trainset = tick_loader
