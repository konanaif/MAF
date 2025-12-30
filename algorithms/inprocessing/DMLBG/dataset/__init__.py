from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .dep import dep
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'dep' : dep,
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
