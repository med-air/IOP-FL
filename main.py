import sys, os
import logging

from torch.utils import data
from torch.utils.data import dataset
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from federated.configs import set_configs
from data.prostate.generate_data import load_partition_data_federated_prostate
from federated.fl_api import FedAvgAPI
from federated.model_trainer_segmentation import ModelTrainerSegmentation



def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     

def set_paths(args):
     args.save_path = '../io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}'.format(args.data, args.target, args.seed, args.batch, args.lr)

     exp_folder = '{}'.format(args.mode)

     if args.balance:
          exp_folder = exp_folder + '_balanced'

     print(exp_folder)
     args.save_path = os.path.join(args.save_path, exp_folder)
     if not os.path.exists(args.save_path):
          os.makedirs(args.save_path)

def custom_model_trainer(args):
     
     args.lr = 1e-3
     if args.ood_test:
         from nets.UNet import UNet
         model = UNet(input_shape=[3, 384, 384])
     else:
         from nets.models import UNet
         model = UNet(input_shape=[3, 384, 384])
     model_trainer = ModelTrainerSegmentation(model, args)
     
     return model_trainer

def custom_dataset(args):
     if args.data == "prostate":
          datasets = load_partition_data_federated_prostate(args)
     return datasets

def custom_federated_api(args, model_trainer, datasets):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     # print(device)
     if args.mode == "fedavg":
          federated_api = FedAvgAPI(datasets, device, args, model_trainer)
     return federated_api
     

if __name__ == "__main__":
     args = set_configs()
     args.generalize = False
     deterministic(args.seed)
     set_paths(args)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
     log_path = args.save_path.replace('checkpoint', 'log')
     if not os.path.exists(log_path): os.makedirs(log_path)
     log_path = log_path+'/log.txt' if args.log else './log.txt'
     logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
     logging.info(str(args))

     model_trainer = custom_model_trainer(args)
     datasets = custom_dataset(args)
     federated_manager = custom_federated_api(args, model_trainer, datasets)
     if args.ood_test:
            
          print('Test time dynamic route')
          from nets.routeconv import RouteConv2D, RouteConvTranspose2D
          
          global_round = {
               'I2CVB': 95,
               'ISBI': 99,
               'HK': 96,
               'BIDMC': 97,
               'UCL': 95,
               'ISBI_1.5': 99,
               }
          
          ckpt = torch.load('./snapshots/{}/fedavg_global_round{}'.format(args.target, global_round[args.target]))
          
          model_trainer.set_model_params(ckpt)
          print('Finish intialization')
          
          rounds = {
               'I2CVB': [55, 73, 89, 92, 79],
               'ISBI': [84, 87, 97, 99, 86],
               'HK': [80, 92, 73, 87, 91],
               'BIDMC': [78, 91, 72, 80, 74],
               'UCL': [99, 90, 87, 91, 99],
               'ISBI_1.5': [50, 83, 99, 99, 94],
               }
          
          paths = [
              torch.load('./snapshots/{}/fedavg_idx_0_round{}'.format(args.target, rounds[args.target][0])),
              torch.load('./snapshots/{}/fedavg_idx_1_round{}'.format(args.target, rounds[args.target][1])),
              torch.load('./snapshots/{}/fedavg_idx_2_round{}'.format(args.target, rounds[args.target][2])),
              torch.load('./snapshots/{}/fedavg_idx_3_round{}'.format(args.target, rounds[args.target][3])),
              torch.load('./snapshots/{}/fedavg_idx_4_round{}'.format(args.target, rounds[args.target][4])),
              torch.load('./snapshots/{}/fedavg_global_round{}'.format(args.target, global_round[args.target]))
          ]
        
          for m in model_trainer.model.modules():
              if isinstance(m, RouteConv2D) or isinstance(m, RouteConvTranspose2D):
                  m._mix_trajectories(paths)
                  
          metrics = federated_manager.ood_client.test_time_adaptation_by_iopfl(None)
          
     elif args.test:
          ckpt = torch.load('trained local trajectory path')
          model_trainer.set_model_params(ckpt)
          test_data_local_dict = datasets[1][-2]
          # test the trajectroy on all local clients
          for client_idx in range(datasets[0]):
               metrics = model_trainer.test(test_data_local_dict[client_idx], federated_manager.device, args)     
               print(metrics)
     else:
          federated_manager.train()
     
     
