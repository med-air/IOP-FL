import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import DiceLoss, entropy_loss
from data.prostate.transforms import transforms_for_noise, transforms_for_rot, transforms_for_scale, transforms_back_scale, transforms_back_rot
import copy
import numpy as np
import random

def smooth_loss(output, d=10):
    
    output_pred = torch.nn.functional.softmax(output, dim=1)
    output_pred_foreground = output_pred[:,1:,:,:]
    m = nn.MaxPool2d(kernel_size=2*d+1, stride=1, padding=d)
    loss = (m(output_pred_foreground) + m(-output_pred_foreground))*(1e-3*1e-3)
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss
    
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)


class ModelTrainerSegmentation(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    @torch.enable_grad()    
    def io_pfl(self, test_data, device, args):
        deterministic(args.seed)
        metrics = {
            'test_dice': 0,
            'test_loss': 0,
        }
        best_dice = 0.
        dice_buffer = []
        model = self.model
        model_adapt = copy.deepcopy(model)
        model_adapt.to(device)
        model_adapt.train()
        for m in model_adapt.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        var_list = model_adapt.named_parameters()
        update_var_list = []
        update_name_list = []
        names = generate_names()

        for idx, (name, param) in  enumerate(var_list):
            param.requires_grad_(False)
            if "bn" in name or name in names:
                param.requires_grad_(True)
                update_var_list.append(param)
                update_name_list.append(name)
        optimizer = torch.optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
        criterion = DiceLoss().to(device)
        loss_all = 0
        test_acc = 0.
        
        for epoch in range(10):
            loss_all = 0
            test_acc = 0.
            
            deterministic(args.seed)
            for step, (data, target) in enumerate(test_data):
                deterministic(args.seed)
                data = data.to(device)
                
                if epoch > -1:
                    
                    input_u1 = copy.deepcopy(data)
                    input_u2 = copy.deepcopy(data)
                    input_u1 = transforms_for_noise(input_u1, 0.5)
                    input_u1, rot_mask, flip_mask = transforms_for_rot(input_u1)
    
                
                
                target = target.to(device)
                output = model_adapt(data, "main")
               
                loss_entropy_before = entropy_loss(output, c=2)
                if epoch > -1:
                    output_u1 = model_adapt(input_u1, "main")
                    output_u2 = model_adapt(input_u2, "main")
                    output_u1 = transforms_back_rot(output_u1, rot_mask, flip_mask)
                    consistency_loss = softmax_mse_loss(output_u1, output_u2)
                loss_smooth = smooth_loss(output)
                loss_smooth = torch.norm(loss_smooth)
                
                    
            
                all_loss = loss_entropy_before
                if epoch > -1:
                    all_loss = 10*all_loss + 0.1*torch.mean(consistency_loss) + loss_smooth 
                                
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                output = model_adapt(data, "main")
                
                loss = criterion(output, target)
                loss_all += loss.item()
            
                test_acc += DiceLoss().dice_coef(output, target).item()

            report_acc = round(test_acc/len(test_data),4)
            print('Test Acc:', report_acc)
           
            if best_dice < test_acc/len(test_data):
                best_dice = test_acc/len(test_data)

            dice_buffer.append(report_acc)
            print('Acc History:', dice_buffer)
            
        
        loss = loss_all / len(test_data)
        acc = test_acc/ len(test_data)
        print('Best Acc:', round(best_dice,4)) 
        # print(dice_buffer)
        metrics['test_loss'] = loss
        metrics["test_dice"] = acc
        return metrics
        
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = DiceLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x, "main")
                loss = criterion(log_probs, labels)
                
                acc = DiceLoss().dice_coef(log_probs, labels).item()

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc)
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args, ood=False):
        model = copy.deepcopy(self.model)

        model.to(device)
        if ood:
            model.train()
        else:
            model.eval()

        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }

        criterion = DiceLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x, "main")
                loss = criterion(pred, target)

                acc = DiceLoss().dice_coef(pred, target).item()

                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_acc'] += acc
        metrics["test_loss"] = metrics["test_loss"] / len(test_data)
        metrics["test_acc"] = metrics["test_acc"] / len(test_data)
        return metrics
    

def generate_names():
    names = []
    for i in range(1, 5):
        for j in range(1, 3):
            name = "encoder{}.enc{}_conv{}._routing_fn.fc1.weight".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc1.bias".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc1.weight".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc1.bias".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc2.weight".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc2.bias".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc2.weight".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc2.bias".format(i,i,j)
            names.append(name)
    names.append('conv._routing_fn.fc1.weight')
    names.append('conv._routing_fn.fc1.bias')
    names.append('conv._routing_fn.fc2.weight')
    names.append('conv._routing_fn.fc2.bias')
    for i in range(1, 5):
        name = "upconv{}._routing_fn.fc1.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc1.bias".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.bias".format(i)
        names.append(name)
    return names
