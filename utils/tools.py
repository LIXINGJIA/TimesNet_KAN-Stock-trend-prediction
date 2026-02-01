import numpy as np
import torch
import math
import csv
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose  # whether to print tip info
        self.counter = 0 # now how many times loss not on decrease
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        # meaning: current score is not 'delta' better than best_score, representing that 
        # further training may not bring remarkable improvement in loss. 
        elif score < self.best_score + self.delta:  
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 'No Improvement' times become higher than patience --> Stop Further Training
            if self.counter >= self.patience:
                self.early_stop = True

        else: #model's loss is still on decrease, save the now best model and go on training
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    ### used for saving the current best model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def cal_precision(y_pred, y_true):
    tp = np.sum((y_pred == 1)& ((y_true == 1)))
    predicted_positive = np.sum(y_pred == 1)
    if predicted_positive == 0 :
        return 0.0
    return tp / predicted_positive

def cal_recall(y_pred, y_true):
    tp = np.sum((y_pred == 1)& ((y_true == 1)))
    actual_positive = np.sum(y_true == 1)
    if actual_positive == 0 :
        return 0.0
    return tp / actual_positive

def cal_f1(y_pred, y_true):
    precision = cal_precision(y_pred, y_true)
    recall = cal_recall(y_pred, y_true)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def write_result(file_name:str,stock=0,acc=0,pre=0,recall=0,f1=0,time=0,flag=1):
    if flag==1:
        with open(file_name,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["date", "acc", "precision", "recall", "f1","time"])
            print("write csv header")
    else:
        with open(file_name,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([stock,acc,pre,recall,f1,time])
            print("write result to {}".format(file_name))
