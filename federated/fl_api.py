import copy
import logging
import random
import sys,os

import numpy as np
import pandas as pd
import torch
from .client import Client


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args
        client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data
        
        self.model_trainer = model_trainer
        # setup clients
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer)
        logging.info("############setup ood clients#############")
        self.ood_client = Client(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)
      
        self.ood_performance = {"before":[]} 
        self.local_performance_by_global_model = dict() 
        self.local_performance_by_trajectory = dict()   
        self.local_val_by_global_model = dict() 
        self.local_val_by_trajectory = dict()   
        self.ood_performance_by_trajectory = dict() 
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_performance_by_trajectory[f'idx{idx}'] = []
            self.ood_performance_by_trajectory[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []
            self.local_val_by_trajectory[f'idx{idx}'] = []

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        # logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("============ Communication round : {}".format(round_idx))

            w_locals = []

            client_indexes = self._client_sampling(round_idx, self.client_num_in_total,
                                                   self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # client.save_trajectory(round_idx)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            # save global weights
            # torch.save(w_global, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))
            self.model_trainer.set_model_params(w_global)
            # local validation
            self._local_val_on_all_clients(round_idx)
            #local test
            self._local_test_on_all_clients(round_idx)
            # test results
            # self._ood_test_on_global_model(round_idx, self.ood_client, self.ood_data, w_global)
            # self._ood_test_on_trajectory(round_idx)
          

            # local val results
            local_val_by_global_model_pd = pd.DataFrame.from_dict(self.local_val_by_global_model)
            local_val_by_trajectory_pd = pd.DataFrame.from_dict(self.local_val_by_trajectory)
            # local test results
            local_performance_by_global_model_pd = pd.DataFrame.from_dict(self.local_performance_by_global_model)
            local_performance_by_trajectory_pd = pd.DataFrame.from_dict(self.local_performance_by_trajectory)
            # ood results
            # ood_performance_pd = pd.DataFrame.from_dict(self.ood_performance)
            # ood_performance_by_trajectory_pd = pd.DataFrame.from_dict(self.ood_performance_by_trajectory)


            local_val_by_global_model_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_local_val_by_global_model.csv"))
            local_val_by_trajectory_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_local_val_by_trajectory.csv"))
            local_performance_by_global_model_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_local_performance_by_global_model.csv"))
            local_performance_by_trajectory_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_local_performance_by_trajectory.csv"))
            # ood_performance_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_ood_performance.csv"))
            # ood_performance_by_trajectory_pd.to_csv(os.path.join(self.args.save_path,self.args.mode + "_ood_performance_by_trajectory.csv"))
           
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes


    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    
    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        ''' unify key'''
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)            
        logging.info(stats)
        return metrics
        
    def _ood_test_on_trajectory(self, round_idx):
        logging.info("============ ood_test_on_all_trajectory : {}".format(round_idx))
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            test_ood_metrics_by_trajectory = client.ood_test_by_trajectory(self.ood_data)
            self.ood_performance_by_trajectory["idx" + str(client_idx)].append(copy.deepcopy(test_ood_metrics_by_trajectory['test_acc']))

    def _local_val_on_all_clients(self, round_idx):
        logging.info("============ local_validation_on_all_clients : {}".format(round_idx))

        val_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            local_metrics = client.local_validate()
            local_metrics_by_trajectory = client.local_validate_by_trajectory()
            
            self.local_val_by_global_model["idx" + str(client_idx)].append(copy.deepcopy(local_metrics['test_acc']))
            self.local_val_by_trajectory["idx" + str(client_idx)].append(copy.deepcopy(local_metrics_by_trajectory['test_acc']))
            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics_by_trajectory['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics_by_trajectory['test_loss'] ))
        # logging.info(val_metrics)

    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))

        test_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # test data
            test_local_metrics = client.local_test(True)
            test_local_metrics_by_trajectory = client.local_test_by_trajectory()
            
            self.local_performance_by_global_model["idx" + str(client_idx)].append(copy.deepcopy(test_local_metrics['test_acc']))
            self.local_performance_by_trajectory["idx" + str(client_idx)].append(copy.deepcopy(test_local_metrics_by_trajectory['test_acc']))
            test_metrics['acc'].append(copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, test_local_metrics['test_acc'], test_local_metrics['test_loss'] ))
        # logging.info(test_metrics)

            
    def test_time_adaptation(self, w_global=None):  
        metrics = self.ood_client.test_time_adaptation_by_iopfl(copy.deepcopy(w_global))
        
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        self.ood_performance["after"].append(test_acc)
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info("############  performance after test time adaptation  #############")    
        logging.info(stats)
        return metrics


    
    

