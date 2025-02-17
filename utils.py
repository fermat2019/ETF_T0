from datetime import datetime
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import pickle
import os
 
matplotlib.rcParams['font.family'] = 'SimHei'  # 指定中文字体为黑体
matplotlib.rcParams['font.size'] = 12  # 设置字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略 SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
np.seterr(divide='ignore', invalid='ignore')



class FactorIC():
    def __init__(self,
                 local_path='.',):
        self.ETF_minquotes = pd.read_parquet(f'{local_path}/ETF_minquotes.parquet')
        self.F588080 = self.ETF_minquotes.loc[self.ETF_minquotes['windcode'] == '588080.SH']
        self.F159915 = self.ETF_minquotes.loc[self.ETF_minquotes['windcode'] == '159915.SZ']
        self.minquotes = {'F588080': self.F588080, 'F159915': self.F159915}
        for Fund_code in ['F588080', 'F159915']:
            self.minquotes[Fund_code].loc[:,'htc_ret'] = self.minquotes[Fund_code].groupby(self.minquotes[Fund_code].index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values
            self.minquotes[Fund_code].loc[:,'h1hour_ret'] = self.minquotes[Fund_code].groupby(self.minquotes[Fund_code].index.date).apply(lambda x:x['close'].iloc[-60]/x['open'].shift(-1)-1).values
        self.ICresult = {'F588080':{},'F159915':{}}

    def CalcFactors(self, FactorFunc):
        for Fund_code in ['F588080', 'F159915']:
            self.minquotes[Fund_code] = FactorFunc(self.minquotes[Fund_code])

    def GetFactorIC(self, Fund_code, factor_name):
        """计算因子的择时IC"""
        #[minquote.index.strftime("%H:%M")<="14:30"].replace((np.inf,-np.inf),0) #可能会因为缺失值导致相关系数计算出异常值
        temp_minquote = self.minquotes[Fund_code][self.minquotes[Fund_code].index.date>self.minquotes[Fund_code].index.date[0]]  # 剔除第一天的数据
        temp_minquote = self.minquotes[Fund_code][self.minquotes[Fund_code].index.strftime("%H:%M")<="14:30"]  # 计算持有至收盘的因子IC
        temp_minquote[factor_name] = temp_minquote[factor_name].rolling(240,min_periods=120).rank(pct=True)  # 计算因子过去一天的排名分位数

        IC = temp_minquote.groupby(temp_minquote.index.strftime("%H:%M")).apply(lambda x:x[f'{factor_name}'].corr(x['htc_ret']))
        self.ICresult[Fund_code][factor_name] = IC
        IC.cumsum().plot(figsize=(15,5), label=f'{factor_name}持有至收盘')
        plt.legend(loc='lower left')
        print(f"{Fund_code}: {factor_name}: 持有至收盘 IC: {IC.mean():.3f}, IR: {IC.mean()/IC.std():.3f}")
        return IC.mean(),IC.mean()/IC.std()

    def GetICResult(self, Fund_code, factor_list):
        """遍历多个因子分别计算IC"""
        result = {}
        for factor in factor_list:
            ICmean, IR = self.GetFactorIC(Fund_code, factor)
            result[factor] = (ICmean, IR)
        result = pd.DataFrame(result,index=['持有至收盘IC','持有至收盘IR']).T
        return result

def CalcFinalFactors(minquote):
    minquote.loc[:,'open0930'] = minquote.groupby(minquote.index.date)['open'].transform(lambda x: x.iloc[0])  # 0930开盘价
    minquote.loc[:,'move'] = (minquote['close']/minquote['open0930']-1).abs()  # 计算每分钟的价格位移
    minquote.loc[:,'move_rolling10'] = minquote.groupby(minquote.index.strftime('%H:%M'))['move'].transform(lambda x: x.rolling(10,min_periods=5).mean())
    minquote.loc[:,'close_last'] = (minquote.groupby(minquote.index.date)['close'].transform(lambda x: x.iloc[-1])).shift(240)  # 前一日收盘价
    minquote.loc[:,'lower_bound'] = minquote[['open0930','close_last']].min(axis=1)*(1-minquote['move_rolling10'])  # 计算下限
    minquote.loc[:,'close_lower_bound'] = minquote['close']/minquote['lower_bound']
    
    minquote.loc[:,'BOLLup_4hour'] = minquote['close'].rolling(240,min_periods=120).mean()+2*minquote['close'].rolling(240,min_periods=120).std()
    minquote.loc[:,'close_BOLLup_4hour'] = minquote['close']/minquote.loc[:,'BOLLup_4hour']

    minquote.loc[:,'EMA_1hour'] = minquote['close'].ewm(span=60,min_periods=30).mean()
    minquote.loc[:,'EMA_4hour'] = minquote['close'].ewm(span=240,min_periods=120).mean()
    minquote.loc[:,'MACD_1hour_4hour'] = minquote['EMA_1hour'] - minquote['EMA_4hour']
    
    minquote.loc[:,'close_diff'] = minquote['close'] - minquote['close'].shift(1)
    minquote.loc[:,'RSI_1hour'] = -(minquote['close_diff']*(minquote['close_diff'] > 0)).rolling(60,min_periods=30).mean() / (minquote['close_diff']*(minquote['close_diff'] < 0)).rolling(60,min_periods=30).mean()
    
    minquote.loc[:,'volume_ratio'] = minquote['volume']/minquote['volume'].rolling(480,min_periods=60).mean()
    minquote.loc[:,'ret_min'] = minquote['close']/minquote['open']-1
    minquote.loc[:,'ret_volume_corr_4hour'] = minquote['ret_min'].rolling(240,min_periods=120).corr(minquote['volume_ratio'])
    
    minquote.loc[:,'volume_std_1hour'] = minquote['volume_ratio'].groupby(minquote.index.date).transform(lambda x:x.rolling(60,min_periods=30).std())
    return minquote

class ETFIntradayStrategy:
    def __init__(self, 
                 data, 
                 factor_name, 
                 direction = 1,
                 periods = [('09:59','10:59'),('13:29','14:29')],
                 rolling_is_rank = True,   # 是否使用滚动排序作为分位数的计算方式
                 rolling_window = 480,  # 滚动排序窗口大小
                 bound_is_series = False,  # upperbound 与 lowerbound是否是序列
                 upper_bound = 0.8,  #分位数上界
                 lower_bound = 0.2,  #分位数下界
                 upper_stop_bound = 0.5,   #多仓止损分位数
                 lower_stop_bound = 0.5,   #空仓止损分位数
                 stop_loss_ratio = 0.02,  #止损比例
                   ):
        """
        :param data: DataFrame, 包含分钟级别的交易数据，index为datetime，至少有open，close，factor_name列(factor_name列已经标准化为0-1）
        :param factor_name: str,因子值的列名
        :param periods: list, 信号作用的时间段
        :param window: int,计算分位数的滚动窗口大小
        :param 
        :param buy_bound: float,买入信号的阈值，这里的阈值可以是一个序列
        :param sell_bound: float,卖出信号的阈值
        """
        self.data = data.copy()
        self.factor_name = factor_name
        self.direction = direction
        self.periods = periods
        self.rolling_is_rank = rolling_is_rank
        self.rolling_window = rolling_window
        self.bound_is_series = bound_is_series
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.upper_stop_bound = upper_stop_bound
        self.lower_stop_bound = lower_stop_bound
        self.stop_loss_ratio = stop_loss_ratio
        self.fee_rate = 0  # 先不考虑手续费率  单边万分之一
        self.generate_position()

    def get_position(self,temp):
        """根据信号生成仓位"""
        OpenTime = temp.loc[temp['signal']!=0,'signal']  # 开仓时刻（下一时刻的open价买入）
        OpenTime = OpenTime.idxmin() if len(OpenTime) != 0 else temp['signal'].idxmin()  # 开仓时刻（下一时刻的open价买入）
        direction = temp.loc[OpenTime,'signal'] #确定一开始的买入卖出方向 1表示买入，-1表示卖出

        temp.loc[:,'Openpos'] = (temp.index>OpenTime)  # 判断是否已开仓
        temp.loc[:,'Openprice'] = temp.loc[temp['Openpos'],'open'].iloc[0]  # 记录开仓价格
        temp.loc[:,'Holdret'] = temp['close']/temp['Openprice']-1  # 计算持仓收益
        temp.loc[:,'Stoploss'] = temp['Holdret']>-self.stop_loss_ratio  # 判断是否触发收益止损比例
        if direction == 1:  #这里还需要乘上有效时间段
            temp.loc[:,'Stopsignal'] = temp[self.factor_name] >= temp[f'{self.factor_name}_upper_stop_bound']  # 判断是否触发信号止损分位数
        elif direction == -1:
            temp.loc[:,'Stopsignal'] = temp[self.factor_name] <= temp[f'{self.factor_name}_lower_stop_bound']  # 判断是否触发信号止损分位数
        else:
            temp.loc[:,'Stopsignal'] = False
        mask = pd.Series(False, index=temp.index)
        for start_time, end_time in self.periods:    # 为每个时间段内的时间点设置掩码为 True
            mask = mask | ((temp.loc[:,'%H:%M'] >= start_time) & (temp.loc[:,'%H:%M']  <= end_time))
        temp.loc[~mask, 'Stopsignal'] = True       # 将不在时间段内的信号设为True
        temp.loc[:,'position'] = direction * temp['Openpos'] * temp['Stoploss'] * temp['Stopsignal']  # 计算持仓信号

        CloseTime = temp.loc[(temp['position']==0)&(temp['position'].shift()==direction),'position']  # 记录平仓时刻
        if len(CloseTime) != 0:
            CloseTime = CloseTime.idxmin()  # 显示平仓时刻
            temp.loc[CloseTime,'position'] = direction  # 平仓时刻后持仓信号为0
            temp.loc[temp.index>CloseTime,'position'] = 0
        return temp

    def generate_position(self):
        # 生成信号
        self.data[self.factor_name] = self.direction * self.data[self.factor_name]
        self.data['%H:%M'] = self.data.index.strftime('%H:%M')
        if self.bound_is_series:
            self.data[f'{self.factor_name}_upper_bound'] = self.upper_bound
            self.data[f'{self.factor_name}_lower_bound'] = self.lower_bound
            self.data[f'{self.factor_name}_upper_stop_bound'] = self.upper_stop_bound
            self.data[f'{self.factor_name}_lower_stop_bound'] = self.lower_stop_bound
        else:
            self.data[f'{self.factor_name}_upper_bound'] = self.data[self.factor_name].rolling(self.rolling_window,min_periods = 240).quantile(self.upper_bound)
            self.data[f'{self.factor_name}_lower_bound'] = self.data[self.factor_name].rolling(self.rolling_window,min_periods = 240).quantile(self.lower_bound)
            self.data[f'{self.factor_name}_upper_stop_bound'] = self.data[self.factor_name].rolling(self.rolling_window,min_periods = 240).quantile(self.upper_stop_bound)
            self.data[f'{self.factor_name}_lower_stop_bound'] = self.data[self.factor_name].rolling(self.rolling_window,min_periods = 240).quantile(self.lower_stop_bound)
            if not self.rolling_is_rank: # 取过去几日的分位数固定当日的上下界
                self.data[f'{self.factor_name}_upper_bound'].shift().groupby(self.data[f'{self.factor_name}_upper_bound'].index.date).transform(lambda x: x.iloc[0])  
                self.data[f'{self.factor_name}_lower_bound'].shift().groupby(self.data[f'{self.factor_name}_lower_bound'].index.date).transform(lambda x: x.iloc[0])
                self.data[f'{self.factor_name}_upper_stop_bound'].shift().groupby(self.data[f'{self.factor_name}_upper_stop_bound'].index.date).transform(lambda x: x.iloc[0])
                self.data[f'{self.factor_name}_lower_stop_bound'].shift().groupby(self.data[f'{self.factor_name}_lower_stop_bound'].index.date).transform(lambda x: x.iloc[0])

        self.data['signal'] = 0
        self.data.loc[self.data[self.factor_name] > self.data[f'{self.factor_name}_upper_bound'], 'signal'] = 1    # 买入信号
        self.data.loc[self.data[self.factor_name] < self.data[f'{self.factor_name}_lower_bound'], 'signal'] = -1   # 卖出信号

        # 创建一个布尔掩码，初始值都为 False
        mask = pd.Series(False, index=self.data.index)
        for start_time, end_time in self.periods:    # 为每个时间段内的时间点设置掩码为 True
            mask = mask | ((self.data.loc[:,'%H:%M'] >= start_time) & (self.data.loc[:,'%H:%M']  <= end_time))
        self.data.loc[~mask, 'signal'] = 0       # 将不在时间段内的信号设置为0
        self.data = self.data.groupby(self.data.index.date).apply(self.get_position).droplevel(0)  # 每日只保留第一个信号

    def calculate_returns(self):
        """计算策略收益（考虑滑点与手续费）"""
        # 计算每日收益
        self.data['o2o_minret'] = self.data.groupby(self.data.index.date)['open'].transform(lambda x: (x.shift(-1) - x) / x)
        self.data['ret'] = self.data['o2o_minret'] * self.data['position']
        self.data['ret'] = self.data['ret'].fillna(0)
        return self.data

    def get_performance(self):
        """计算策略绩效"""
        self.data = self.calculate_returns()
        dailyperformance = pd.DataFrame()
        dailyperformance['direction'] = self.data['position'].groupby(self.data.index.date).apply(lambda x: x[x != 0].iloc[0] if len(x[x!=0])>0 else 0)
        dailyperformance['holdingmins'] = self.data['position'].groupby(self.data.index.date).apply(lambda x: (x != 0).sum())
        dailyperformance['dailyret'] = self.data['ret'].groupby(self.data.index.date).apply(lambda x: (x + 1).prod())-self.fee_rate*(dailyperformance['holdingmins']>0)-1
        print(f"平均持仓时间: {dailyperformance['holdingmins'].mean():2f} 分钟")
        print(f"胜率：{(dailyperformance['dailyret']>0).sum()/(dailyperformance['holdingmins']>0).sum():2f}")
        print(f"年化收益率：{(dailyperformance['dailyret']+1).prod()**(252/len(dailyperformance['dailyret']))-1:3f}")
        print(f"年化夏普比率：{((dailyperformance['dailyret']+1).prod()**(252/len(dailyperformance['dailyret']))-1)/dailyperformance['dailyret'].std()/np.sqrt(252):3f}")
        (dailyperformance['dailyret']+1).cumprod().plot(figsize=(15, 5))
        return dailyperformance
