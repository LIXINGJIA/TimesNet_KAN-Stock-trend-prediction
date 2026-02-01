import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from argparse import Namespace
from layers.kanlayer import KANLayer

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)  #将列表res中存储的k个周期的特征结果，沿着新的最后一个维度堆叠成一个完整的张量
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)  #将权重归一化
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1) #扩展权重维度，使其与待融合的特征张量res形状匹配
        res = torch.sum(res * period_weight, -1)  #按权重（融合进时间序列中）合并k个周期的特征，得到综合了所有显著周期信息的最终特征。
        # residual connection
        res = res + x #残差链接的关键是，将残差连接后的结果与原始输入x进行加和，从而实现残差学习。
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)]) #重复执行e_layers使得moedl中包含多个时间区块这句话的本质是：根据配置的层数e_layers，创建一个包含多个TimesBlock的可训练层列表，作为模型的深度特征提取器。后续在forward中，会按顺序调用这些TimesBlock（如for i in range(self.layer): enc_out = self.model[i](enc_out)），实现对时间序列的多层级周期特征学习。
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        # self.projection = nn.Linear(
        #         configs.d_model * configs.seq_len, configs.num_class)
        self.projection = KANLayer(configs.d_model * configs.seq_len, configs.num_class,num=5,k=3,device=configs.device,save_plot_data = False,sparse_init=False)
#         DataEmbedding是一个自定义的嵌入层类（专为时间序列设计），核心功能是融合 “数值特征” 和 “时间信息”，并将其映射到高维空间（增强特征表达能力）。具体来说，它通常包含以下子模块（根据实现可能略有差异）：
# 数值特征嵌入：将原始输入的数值特征（如股价、温度等）从低维（enc_in）映射到高维（d_model）；
# 时间特征嵌入：将时间戳信息（如小时、日、月等）转换为向量（捕捉时间周期性，如每天 9 点的规律）；
# 融合与 dropout：将数值嵌入和时间嵌入相加 / 拼接，并通过 dropout 层防止过拟合。
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)#这句话的作用是初始化一个 “层归一化（Layer Normalization）” 层，用于对模型中的高维特征进行归一化处理，稳定训练过程并提升特征学习的效率。通常在嵌入层之后，通常会跟随一个 “层归一化” 层，以 stabilize 训练过程并提升特征学习的效率。


    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output,_,_,_ = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]




if __name__ == '__main__':
    configs=Namespace(
        task_name="classification",
        label_len=15,#is work?
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
        device="cpu",
    )
    model=Model(configs)