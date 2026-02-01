# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
#     MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_custom
import os,sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 拼接父目录（假设模块在当前脚本的上层目录）
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# print(os.getcwd())
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

from data_provider.data_loader import Data_stock
from argparse import Namespace
# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'custom': Dataset_Custom,
#     'm4': Dataset_M4,
#     'PSM': PSMSegLoader,
#     'MSL': MSLSegLoader,
#     'SMAP': SMAPSegLoader,
#     'SMD': SMDSegLoader,
#     'SWAT': SWATSegLoader,
#     'UEA': UEAloader
#     'custon': Dataset_Custom,
# }

    


data_dict = {
    "stock": Data_stock,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'val') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # if args.task_name == 'anomaly_detection':
    #     drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         win_size=args.seq_len,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            collate_fn=collate_fn,
            pin_memory=True,  # 锁页内存，Ubuntu下无兼容问题，GPU数据传输提速5-10倍
            prefetch_factor=2,  # 提前加载2批数据，GPU算完即取，无空闲
            persistent_workers=True,  # 持久化加载进程，避免每个epoch重建进程（PyTorch1.11.0支持完美）
        )
        return data_set, data_loader
    # else:
    #     if args.data == 'm4':
    #         drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         features=args.features,
    #         target=args.target,
    #         timeenc=timeenc,
    #         freq=freq,
    #         seasonal_patterns=args.seasonal_patterns
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
    

