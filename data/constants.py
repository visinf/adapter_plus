NUM_CLASSES_VTAB = {
    'caltech101': 102,
    'cifar100': 100,
    'dtd': 47,
    'flowers102': 102,
    'pet': 37,
    'camelyon': 2,
    'sun397': 397,
    'svhn': 10,
    'resisc45': 45,
    'eurosat': 10,
    'dmlab': 6,
    'kitti': 4,
    'smallnorb_azimuth': 18,
    'smallnorb_elevation': 9,
    'dsprites_position': 16,
    'dsprites_orientation': 16,
    'clevr_distance': 6,
    'clevr_count': 8,
    'retinopathy': 5,
}

NUM_CLASSES_FGVC = {
    "cub": 200,
    "nabirds": 555,
    "oxfordflower": 102,
    "stanfordcars": 196,
    "stanforddogs": 120,
}

NUM_CLASSES = {**NUM_CLASSES_VTAB, **NUM_CLASSES_FGVC}