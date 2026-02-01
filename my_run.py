import numpy as np
from argparse import Namespace
import torch
import random
from classification import Exp_Classification #
from utils.tools import write_result
import os
import pandas as pd
def train(save_path:str,data_path:str):
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args= Namespace(
#核心类
        task_name="classification",
        label_len=15,
        seq_len=50,
        pred_len=0,
        top_k=3,
        d_ff=256,
        num_kernels=3,
        e_layers=3,
        enc_in=6,
        d_model=64,
        embed="timeF",
        freq="d",
        dropout=0.1,
        num_class=2,
#训练类
        learning_rate=0.0001,
        train_epochs=50,
        # model="TimesNet",
        model="KAN",
        use_multi_gpu=False,
        use_gpu=1,
        patience=10,

        is_train=1,
        gpu=0,
        device="",
        itr=1, #训练次数
        gpu_type="cuda",
        model_id="test",
        feature="MS",
#数据类
        batch_size=16,
        data="stock",
        augmentation_ratio=0,
        root_path="data/",
        data_path=data_path,
        num_workers=1,

        checkpoints="train/",
        result_file_path=save_path,


    )

    #是否使用gpu或mps
    if torch.cuda.is_available() and args.use_gpu:
        args.device= torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    Exp = Exp_Classification
    if args.is_train:
        for ii in range(args.itr):
            exp=Exp(args)
            setting="first_train"
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache() 
    else:
        pass
if __name__ == '__main__':
    save_path = "pre_result.csv"
    data=pd.read_csv(save_path)
    data=data["date"].tolist()
    data=[x for x in data if x != "date"]
    write_result(save_path,flag=1)
    folder="data/"
    for csv_file in os.listdir(folder):
        if csv_file[:-4]  in data:
            print(csv_file[:-4]+"is done")
            continue
        else:
            train(save_path,csv_file)          
