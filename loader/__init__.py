import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from config import cfg

from loader.cifar10 import get_cifar10
from loader.cifar100 import get_cifar100
from loader.imagenet import get_imagenet
from loader.imagenette import get_imagenette

def get_loader():
    pair = {
        'cifar10': get_cifar10,
        'cifar100': get_cifar100,
        'imagenet': get_imagenet,
        'imagenette': get_imagenette
    }

    return pair[cfg['data']['type']]()
