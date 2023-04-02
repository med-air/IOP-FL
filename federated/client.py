import logging
import torch
import os
import copy

class Client:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.trajectory = None
        self.prev_weight = None

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.calcuate_trajectory(weights)
        self.prev_weight = weights
        return weights

    def local_validate(self, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    def local_test(self, b_use_test_dataset, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)

        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics
    
    def local_test_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    def local_validate_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_val_data, self.device, self.args)
        del model_trainer_copy
        return metrics


    def ood_test_by_trajectory(self, ood_test_data):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(ood_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics


    def save_trajectory(self, comm_round):
        torch.save(self.trajectory, os.path.join(self.args.save_path, "{}_idx_{}_round{}".format(self.args.mode, self.client_idx, comm_round)))
    
    def calcuate_trajectory(self, w_local):
        if self.trajectory == None:
            self.trajectory = w_local        
        else:
            for k in w_local.keys():
                self.trajectory[k] = self.args.alpha * self.trajectory[k] + (1-self.args.alpha) * w_local[k]
                
    def test_time_adaptation_by_iopfl(self, w_global):
        if w_global != None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
        