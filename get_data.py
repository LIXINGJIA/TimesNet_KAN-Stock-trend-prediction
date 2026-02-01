import tushare as ts
import pandas as pd
import os
token='0aaadebd3d92f786adbe10d9dd7fbdb98ce41a9ef8a00ee66008a34a'
pro = ts.pro_api(token)
def get_data(stock,path):

    path=os.path.join(path,stock+'.csv')
    df = pro.daily(**{
        "ts_code":stock ,
        "trade_date": "",
        "start_date": 20220101,#2020还是2022
        "end_date": 20240201,
        "limit": "",
        "offset": ""
    }, fields=[
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ])
    df=df.sort_values("trade_date")
    df.insert(df.shape[1],"updown",0)
    for i in range(df.shape[0]-1):
        today_close = df.iloc[i]["close"]  # 今天的收盘价
        tomorrow_close = df.iloc[i+1]["close"]  # 明天的收盘价
        if today_close <= tomorrow_close:
            # 今天的标签设为1（涨或平）
            df.iloc[i, df.columns.get_loc("updown")] = 1
    df = df.iloc[:-1]
    df=df.rename(columns={"trade_date":"date"})
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    print(df.head())
    df.to_csv(path,index=False)

def get_stock_codes(num=500):
    """获取指定数量的A股股票代码（上交所+深交所）"""
    # 调用stock_basic接口获取股票列表
    # is_hs: 是否沪深港通标的（N否 H沪股通 S深股通），这里选所有A股
    # list_status: 上市状态（L上市 D退市 P暂停上市），选L
    df = pro.stock_basic(
        exchange='',
        list_status='L',  # 只取上市股票
        fields='ts_code, name'
    )
    
    # 筛选A股（排除B股、北交所等，A股代码以60、00、30开头）
    a_share_codes = []
    for code in df['ts_code']:
        if code.startswith(("30")):  # 上交所A股(60)、深交所A股(00/30)'60', '00', '30'
            a_share_codes.append(code)
        if len(a_share_codes) >= num:
            break  # 取够指定数量即停止
    
    return a_share_codes
if __name__ == '__main__':
    code=get_stock_codes(num=10)
    for i in code:   
        root_path="data/"
        get_data(stock=i,path=root_path)
    # df=pd.read_csv('data/')
    # get_data(stock='000001.SZ',path='data/')