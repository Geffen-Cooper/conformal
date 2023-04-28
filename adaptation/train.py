'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
from tqdm import tqdm
import time

def train(model,train_loader,val_loader,device,lr,epochs,grad_accum=None,log_name="log"):
    
    # init tensorboard
    writer = SummaryWriter()

    # create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=0.00002,momentum=0.9)
    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.2)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    # iters = len(train_loader)

    best_val_acc = 0

    model.train()
    model = model.to(device)
    batch_iter = 0

    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # print("train, num samples:", len(target))
            # Big Forward
            data, target = data.to(device), target.to(device)

            if grad_accum != None:
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss = loss/grad_accum
                writer.add_scalar("Loss/train", loss, batch_iter)
                loss.backward()
                if ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
            else:
                optimizer.zero_grad()
                output = model(data)
                # print(output)
                # print(target)
                loss = nn.CrossEntropyLoss()(output, target)
                writer.add_scalar("Loss/train", loss, batch_iter)
                loss.backward()
                optimizer.step()
                

            if batch_idx % 800 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, lr: {:.8f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss, scheduler1.get_last_lr()[0]))
                if batch_idx > 0:
                    scheduler1.step()
            batch_iter+=1
        scheduler2.step()
        # scheduler.step()
        # evaluate on all the validation sets
        val_acc, val_loss, top5 = validate(model,val_loader,device)
        writer.add_scalar("Loss/val", val_loss, batch_iter)
        writer.add_scalar("Accuracy/val", val_acc, batch_iter)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f}, top5: {:.3f}, lr: {:.8f}'.format(
            e, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss, val_loss, val_acc, top5, scheduler1.get_last_lr()[0]))
        # scheduler1.step()
        if best_val_acc < val_acc:
            print("==================== best validation accuracy ====================")
            print("epoch: {}, val accuracy: {}".format(e,val_acc))
            best_val_acc = val_acc
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
                'lr': scheduler1.get_last_lr(),
                }, 'best_batch_i'+str(batch_iter)+log_name+str(time.time())+'.pth')
        # scheduler2.step()

        # evaluate on all the validation sets
        # for idx,val_loader in enumerate(val_loaders):
        #     val_acc, val_loss = validate(model, val_loader,args)
        #     # writer.add_scalar("Loss/val", val_loss, batch_iter)
        #     # writer.add_scalar("Accuracy/val", val_acc, batch_iter)
        #     val_losses[idx].append(val_loss)
        #     val_accs[idx].append(val_acc)
        #     val_iters[idx].append(batch_iter)

        # Save the best validation accuracy and the corresponding model.
    #     if best_val_acc < val_acc:
    #         best_val_acc = val_acc
    #         torch.save({
    #             'epoch': e+1,
    #             'model_state_dict': model.state_dict(),
    #             'val_acc': best_val_acc,
    #             'train_loss':train_losses,
    #             'train_iter':train_iters,
    #             'val_loss': val_loss,
    #             'val_accs':val_accs,
    #             'val_iter':val_iters
    #             }, str(args.log_name)+'.pth')

    #     # print the 
    #     print('In epoch {}, train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
    #         e, loss, val_loss, val_acc, best_val_acc))
    #     # scheduler.step()
    
    # return model,train_losses,train_iters,val_losses,val_accs,val_iters


    
def validate(model, val_loader, device):
    model.eval()
    model = model.to(device)
    val_loss = 0
    correct = 0
    correct5 = 0

    with torch.no_grad():
        # confusion_matrix = torch.zeros(200, 200)
        i=0
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

            batch_size = target.size(0)

            _, pred_top5 = torch.topk(output, 5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(target.view(1, -1).expand_as(pred_top5))

            correct5 += correct_top5[:5].reshape(-1).float().sum(0).item()
            i+=1
            # print(i*batch_size)
            # print(pred)
            # for t, p in zip(target.view(-1), pred.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
                
        # print(confusion_matrix)

        # Compute loss and accuracy
        val_loss /= len(val_loader) # cross entropy returns batch mean
        val_acc = correct / len(val_loader.dataset)
        val_acc_top5 = correct5 / len(val_loader.dataset)
        return val_acc, val_loss, val_acc_top5


def validate_ensemble(models, val_loader, device):
    for i,m in enumerate(models):
        models[i] = models[i].to(device)
        models[i].eval()
    val_loss = 0
    correct = 0
    correct5 = 0
    out_shape = models[0](next(iter(val_loader))[0][0].unsqueeze(0).to(device)).shape[1]
    with torch.no_grad():
        # confusion_matrix = torch.zeros(200, 200)
        i=0
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            confidences = torch.zeros((len(target),out_shape)).to(device)
            n = len(models)
            for i,m in enumerate(models):
                # confidences += F.softmax(m(data),dim=1)*((i+1)/(0.5*(n*(n+1))))
                confidences += F.softmax(m(data),dim=1)
            confidences /= len(models)

            pred = confidences.argmax(dim=1, keepdim=True)
            val_loss += nn.CrossEntropyLoss()(confidences, target).item()  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

            batch_size = target.size(0)

            _, pred_top5 = torch.topk(confidences, 5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(target.view(1, -1).expand_as(pred_top5))

            correct5 += correct_top5[:5].reshape(-1).float().sum(0).item()
            i+=1
            # print(i*batch_size)
            # print(pred)
            # for t, p in zip(target.view(-1), pred.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
                
        # print(confusion_matrix)

        # Compute loss and accuracy
        val_loss /= len(val_loader) # cross entropy returns batch mean
        val_acc = correct / len(val_loader.dataset)
        val_acc_top5 = correct5 / len(val_loader.dataset)
        return val_acc, val_loss, val_acc_top5




def train_sub(big_model,small_model,train_loader,val_loader,device):
    
    big_model = big_model.to(device)
    small_model = small_model.to(device)

    # init tensorboard
    # big_writer = SummaryWriter()
    # small_writer = SummaryWriter()
    writer = SummaryWriter()

    # create the optimizer
    big_optimizer = torch.optim.SGD(big_model.parameters(), lr=0.01, momentum=0.9)
    small_optimizer = torch.optim.SGD(small_model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_acc = 0

    big_model.train()
    small_model.train()
    batch_iter = 0

    # train_losses = []
    # val_losses = [[] for i in range(len(val_loaders))]
    # val_accs = [[] for i in range(len(val_loaders))]
    # train_iters = []
    # val_iters = [[] for i in range(len(val_loaders))]

    for e in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Big Forward
            data, target = data.to(device), target.to(device)

            big_optimizer.zero_grad()
            big_output = big_model(data)
            big_loss = nn.CrossEntropyLoss()(big_output, target)
            big_loss.backward()
            big_optimizer.step()

            small_idxs = sum([target == i for i in range(200)]).nonzero().view(-1)
            # print(small_idxs)
            small_target = target[small_idxs]
            # print(small_target)
            small_data = data[small_idxs]
            # print(small_data.size())
            
            small_optimizer.zero_grad()
            small_output = small_model(small_data)
            small_loss = nn.CrossEntropyLoss()(small_output, small_target)
            # print(small_loss)
            small_loss.backward()
            small_optimizer.step()

            if batch_idx % 100 == 0:
                # evaluate on all the validation sets
                big_val_acc, big_val_loss = validate_sub(big_model,val_loader,device)
                small_val_acc, small_val_loss = validate_sub(small_model,val_loader,device)
                # big_writer.add_scalar("Loss/val", val_loss, batch_iter)
                # small_writer.add_scalar("Accuracy/val", val_acc, batch_iter)
                writer.add_scalars('ValidationLoss', {
                    'BIG': big_val_loss,
                    'small': small_val_loss,
                }, batch_iter)
                writer.add_scalars('ValidationAcc', {
                    'BIG': big_val_acc,
                    'small': small_val_acc,
                }, batch_iter)
                # val_losses[idx].append(val_loss)
                # val_accs[idx].append(val_acc)
                # val_iters[idx].append(batch_iter)

                # print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f})'.format(
                #     e, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss, val_loss, val_acc, best_val_acc))

            batch_iter+=1

        # evaluate on all the validation sets
        # for idx,val_loader in enumerate(val_loaders):
        #     val_acc, val_loss = validate(model, val_loader,args)
        #     # writer.add_scalar("Loss/val", val_loss, batch_iter)
        #     # writer.add_scalar("Accuracy/val", val_acc, batch_iter)
        #     val_losses[idx].append(val_loss)
        #     val_accs[idx].append(val_acc)
        #     val_iters[idx].append(batch_iter)

        # Save the best validation accuracy and the corresponding model.
    #     if best_val_acc < val_acc:
    #         best_val_acc = val_acc
    #         torch.save({
    #             'epoch': e+1,
    #             'model_state_dict': model.state_dict(),
    #             'val_acc': best_val_acc,
    #             'train_loss':train_losses,
    #             'train_iter':train_iters,
    #             'val_loss': val_loss,
    #             'val_accs':val_accs,
    #             'val_iter':val_iters
    #             }, str(args.log_name)+'.pth')

    #     # print the 
    #     print('In epoch {}, train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
    #         e, loss, val_loss, val_acc, best_val_acc))
    #     # scheduler.step()
    
    # return model,train_losses,train_iters,val_losses,val_accs,val_iters


    
def validate_sub(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        num_samples = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            sub_idxs = sum([target == i for i in range(200)]).nonzero().view(-1)
            sub_target = target[sub_idxs]
            sub_data = data[sub_idxs]
            num_samples += len(sub_target)

            # Forward
            output = model(sub_data)
            pred = output.argmax(dim=1, keepdim=True)
            val_loss += nn.CrossEntropyLoss(reduction='sum')(output, sub_target).item()  # sum up batch loss
            correct += pred.eq(sub_target.view_as(pred)).sum().item()

        # Compute loss and accuracy
        val_loss /= num_samples
        val_acc = correct / num_samples
        # print(num_samples)
        return val_acc, val_loss


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument('--model', type=str, default='mobilenetv2', help='model architecture: mobilenetv2')
    parser.add_argument('--dataset', type=str, default = 'tinyimagenet', help='dataset name: tinyimagenet')
    parser.add_argument('--log_name', type=str, default = 'default', help='checkpoint file name')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--permuted', action='store_true', default=False,
                        help='Load permuted loaders as well')


    args = parser.parse_args()
    print(args)
    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    print("=================")
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    args.device = device

    train(args)