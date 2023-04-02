import torch
from torch.utils.data.dataloader import DataLoader

def generate_data_loader(args, client_num, trainsets, valsets, testsets, ood_set):
     train_data_local_num_dict = dict()
     train_data_local_dict = dict()
     val_data_local_dict = dict()
     test_data_local_dict = dict()
     train_data_num = 0
     val_data_num = 0
     test_data_num = 0
     min_data_len = min([len(s) for s in trainsets])
     if args.balance:
          print(f'Balance training set, using {args.percent*100}% training data')          
     for idx in range(len(trainsets)):
          if args.balance:
               trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len*args.percent))))
          else:
               trainset = trainsets[idx]
          valset = valsets[idx]
          testset = testsets[idx]
          # for debug
          if args.debug:
               trainset = torch.utils.data.Subset(trainsets[idx], list(range(1000)))
               valset = torch.utils.data.Subset(valsets[idx], list(range(1000)))
               testset = torch.utils.data.Subset(testsets[idx], list(range(1000)))
               

          #print(f'[Client {sites[idx]}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
          train_data_num += len(trainset)
          val_data_num += len(valset)
          test_data_num += len(testset)
          train_data_local_num_dict[idx] = len(trainset)
          train_data_local_dict[idx] = DataLoader(trainset, batch_size=args.batch, shuffle=True)
          val_data_local_dict[idx]   = DataLoader(valset, batch_size=args.batch, shuffle=False)
          test_data_local_dict[idx]  = DataLoader(testset, batch_size=args.batch, shuffle=False)
     if args.debug:
          ood_set = torch.utils.data.Subset(ood_set, list(range(1000)))
     ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch, shuffle=False)
     # print(len(ood_loader.dataset))

     return (client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, ood_loader])