from torch.utils.data import dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch

from .dataset import Prostate
from data.generate_data_loader import generate_data_loader

def load_partition_data_federated_prostate(args):
    sites = args.source
    ood_site = args.target if args.target is not None else 'HK'  # for inside training, using a placeholder
    client_num = len(sites)
    
    transform = None
    trainsets = []
    valsets = []
    testsets = []
    
    ood_set = torch.utils.data.ConcatDataset([Prostate(site=ood_site, split='train', transform=transform),
                                            Prostate(site=ood_site, split='val', transform=transform),
                                            Prostate(site=ood_site, split='test', transform=transform)])
    for site in sites:
        trainset = Prostate(site=site, split='train', transform=transform)
        valset = Prostate(site=site, split='val', transform=transform)
        testset = Prostate(site=site, split='test', transform=transform)
        print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)

    dataset = generate_data_loader(args, client_num, trainsets, valsets, testsets, ood_set)

    return dataset
