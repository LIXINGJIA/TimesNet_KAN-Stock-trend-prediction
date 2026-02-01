import torch 
import torch.nn as nn
from layers.kanlayer import KANLayer
from argparse import Namespace

class Model(nn.Module):
    def __init__(self,configs):
        super(Model,self).__init__()
        self.device=configs.device
        self.kan =KANLayer(in_dim=configs.enc_in*configs.seq_len,out_dim=configs.num_class,num=5,k=3,device=self.device,save_plot_data = False,sparse_init=False)
    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc.reshape(x_enc.shape[0], -1)
        output,_,_,_=self.kan(x_enc)
        return output
if __name__=="__main__":
    configs=Namespace(
        device="cpu",
        num_class=2,
        enc_in=3,
    )
    model=Model(configs)
