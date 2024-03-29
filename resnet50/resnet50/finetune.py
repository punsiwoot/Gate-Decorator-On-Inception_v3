
# coding: utf-8

# In[ ]:


''' setting before run. every notebook should include this code. '''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import sys

_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('GGAP')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)

from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "resnet50_imagenet_finetune",
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
        "nesterov": True
    },
    "data": {
        "type": "imagenet",
        "shuffle": True,
        "batch_size": 256,
        "test_batch_size": 512,
        "num_workers": 32
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "finetune_epoch": 60
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
from prune.utils import analyse_model, DoRealPrune


# In[ ]:


set_seeds()
pack = recover_pack()


# In[ ]:


GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
    gbn.extract_from_bn()


# -------
# 
# #### 60% flops reduced

# In[ ]:


model_dict = torch.load('./logs/0506_lower_sparse/60.ckp', map_location='cpu' if not cfg.base.cuda else 'cuda')
pack.net.module.load_state_dict(model_dict)


# In[ ]:


_ = Conv2dObserver.transform(pack.net.module)
pack.net.module.backbone.fc = FinalLinearObserver(pack.net.module.backbone.fc)
tmp = pack.train_loader.batch_sampler.batch_size
pack.train_loader.batch_sampler.batch_size = 32
Meltable.observe(pack, 0.001)
pack.train_loader.batch_sampler.batch_size = tmp
Meltable.melt_all(pack.net)


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


def finetune(pack, T, mute=False):
    logs = []
    epoch = 0

    LR = [1e-2 for i in range(36)] + [1e-3 for i in range(12)] + [1e-4 for i in range(6)] + [1e-5 for i in range(10)]

    for i in range(T):
        for g in pack.optimizer.param_groups:
            g['lr'] = LR[i]
            g['momentum'] = 0.9

        info = pack.trainer.train(pack)
        info.update(pack.trainer.test(pack, topk=(1, 5)))
        info.update({'LR': pack.optimizer.param_groups[0]['lr']})
        epoch += 1
        if not mute:
            print(info)
        logs.append(info)

    return logs


# In[ ]:


pack.optimizer = optim.SGD(
    pack.net.parameters(),
    lr=1,
    momentum=cfg.train.momentum,
    weight_decay=cfg.train.weight_decay,
    nesterov=cfg.train.nesterov
)


# In[ ]:


_ = finetune(pack, T=cfg.gbn.finetune_epoch)
torch.save(pack.net.module.state_dict(), './logs/A_0506_finetune_40.ckp')
