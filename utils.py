from datetime import datetime
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import pickle
import os


matplotlib.rcParams['font.family'] = 'SimHei'  # 指定中文字体为黑体
matplotlib.rcParams['font.size'] = 12  # 设置字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略 SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


def FactorIC(minquote,factor_name):
    """计算因子的择时IC"""
    minquote.loc[:,'htc_ret'] = minquote.groupby(minquote.index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values
    minquote.loc[:,'h1hour_ret'] = minquote.groupby(minquote.index.date).apply(lambda x:x['open'].shift(-60)/x['open'].shift(-1)-1).values
    #[minquote.index.strftime("%H:%M")<="14:30"].replace((np.inf,-np.inf),0) #可能会因为缺失值导致相关系数计算出异常值
    temp_minquote = minquote[minquote.index.strftime("%H:%M")<="14:00"]  # 计算持有至收盘的因子IC
    IC1 = temp_minquote.groupby(temp_minquote.index.date).apply(lambda x:x[f'{factor_name}'].corr(x['htc_ret']))
    IC2 = temp_minquote.groupby(temp_minquote.index.date).apply(lambda x:x[f'{factor_name}'].corr(x['h1hour_ret']))
    IC1.cumsum().plot(figsize=(15,5), label='持有至收盘')
    IC2.cumsum().plot(figsize=(15,5), label='持有一小时')
    plt.legend()
    print(f"持有至收盘 IC: {IC1.mean():.3f}, IR: {IC1.mean()/IC1.std():.3f}")
    print(f"持有一小时 IC: {IC2.mean():.3f}, IR: {IC2.mean()/IC2.std():.3f}")
    return IC1.mean(),IC2.mean()


class ETFIntradayStrategy:
    def __init__(self, data, factor_name, 
                 LimitOpenTime = '09:30',
                 window = 240, 
                 BuyBound = 0.8, 
                 SellBound = 0.2):
        """
        :param data: DataFrame，包含分钟级别的交易数据，index为datetime，至少有open，close，factor_name列(factor_name列已经标准化为0-1）
        :param factor_name: str，因子值的列名
        :param window: int，计算分位数的滚动窗口大小
        :param buy_bound: float，买入信号的阈值，这里的阈值可以是一个序列
        :param sell_bound: float，卖出信号的阈值
        """
        self.data = data
        self.factor_name = factor_name
        self.window = window
        self.BuyBound = BuyBound
        self.SellBound = SellBound
        self.LimitOpenTime = LimitOpenTime
        self.fee_rate = 0  # 先不考虑手续费率  单边万分之一

    def get_position(self,temp):
        """根据信号生成仓位"""
        temp = temp.replace(0,np.nan).ffill().replace(np.nan,0)  # 填充信号
        temp_diff = temp.diff()  # 之后是去找到每天开仓后那一段0的时间段，即为持仓期
        temp_diff.iloc[0] = temp.iloc[0]
        temp_diff = temp_diff.replace(0,np.nan).ffill().replace(np.nan,0)  # 填充信号   目的是保留那些为1或-1的值，其index的下一期则为持仓期
        position = pd.DataFrame(0, index=temp.index, columns=['position'])
        if len(temp_diff[temp_diff.abs()==1])>0:
            position.loc[temp_diff[temp_diff.abs().shift()==1].index] = temp_diff[temp_diff.abs()==1].iloc[0]
        # 可以考虑加一些限制条件，比如不能在最后30分钟开仓

        return position

    def generate_position(self):
        """生成持仓（1表示持有多仓，-1表示持有空仓）"""
        # 生成信号
        signals = pd.DataFrame(0, index=self.data.index, columns=['signal'])
        signals.loc[self.data[self.factor_name] > self.BuyBound, 'signal'] = 1    # 买入信号
        signals.loc[self.data[self.factor_name] < self.SellBound, 'signal'] = -1   # 卖出信号
        # 限制开仓时间
        signals.loc[signals.index.strftime('%H:%M') < self.LimitOpenTime] = 0
        signals['pos'] = signals['signal'].groupby(signals.index.date).apply(self.get_position).values
        return signals

    def calculate_returns(self):
        """计算策略收益（考虑滑点与手续费）"""
        # 合并仓位信号数据
        self.positions = self.generate_position()
        self.data = pd.concat([self.data, self.positions], axis=1)
        # 计算每日收益
        self.data['o2o_minret'] = self.data.groupby(self.data.index.date)['open'].transform(lambda x: (x.shift(-1) - x) / x)
        self.data['ret'] = self.data['o2o_minret'] * self.data['pos']
        self.data['ret'] = self.data['ret'].fillna(0)
        return self.data

    def get_performance(self):
        """计算策略绩效"""
        self.data = self.calculate_returns()
        dailyperformance = pd.DataFrame()
        dailyperformance['direction'] = self.data['pos'].groupby(self.data.index.date).apply(lambda x: x[x != 0].iloc[0] if len(x[x!=0])>0 else 0)
        dailyperformance['holdingmins'] = self.data['pos'].groupby(self.data.index.date).apply(lambda x: (x != 0).sum())
        dailyperformance['dailyret'] = self.data['ret'].groupby(self.data.index.date).apply(lambda x: (x + 1).prod())-self.fee_rate*(dailyperformance['holdingmins']>0)-1
        print(f"平均持仓时间: {dailyperformance['holdingmins'].mean():2f} 分钟")
        print(f"胜率：{(dailyperformance['dailyret']>0).sum()/(dailyperformance['holdingmins']>0).sum():2f}")
        print(f"年化收益率：{(dailyperformance['dailyret']+1).prod()**(252/len(dailyperformance['dailyret']))-1:3f}")
        print(f"年化夏普比率：{((dailyperformance['dailyret']+1).prod()**(252/len(dailyperformance['dailyret']))-1)/dailyperformance['dailyret'].std()/np.sqrt(252):3f}")
        (dailyperformance['dailyret']+1).cumprod().plot(figsize=(15, 5))
        return dailyperformance
