from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DictToObj:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)

def make_dataset(F588080_temp, F159915_temp, seq_len, factor_list, label_name, FillNAN = False, seq_gap = 1):
    ## 这里可能需要考虑填充nan值
    data = []
    label = []
    for Fund_code, Fund_temp in zip(['F588080.SH', 'F159915.SZ'],[F588080_temp, F159915_temp]):
        Fund_temp = Fund_temp[Fund_temp.index.strftime('%H:%M')<='14:29']
        Fund_temp[label_name] = Fund_temp[label_name].apply(lambda x: 0 if x<=0 else(1 if x>0 else np.nan))
        for i in tqdm(range(seq_len, len(Fund_temp), seq_gap)):
            ## 考虑每个特征时序相对位置的变化，即归一化操作
            #temp_data = Fund_temp.iloc[i-seq_len:i][factor_list].rank(axis=0,pct=True).values
            temp_data = Fund_temp.iloc[i-seq_len:i][factor_list].values  #不做归一化，每个因子都是过去一段时间的分位数
            temp_label = Fund_temp.iloc[i-1][label_name]  #判断大于0还是小于0
            
            if FillNAN:
                temp_data[np.isnan(temp_data)] = 0.5   #用排序中间值0.5填充nan值
                data.append(temp_data) 
                label.append((Fund_temp.iloc[i-1].name,Fund_code,temp_label))          
            elif np.isnan(temp_data).sum()+np.isnan(temp_label) == 0:
                data.append(temp_data)
                label.append((Fund_temp.iloc[i-1].name,Fund_code,temp_label))
            else:
                continue
    return np.array(data), pd.DataFrame(label,columns=['time','Fund_code','htc_ret'])

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
    
    minquote.loc[:,'ret_min_abs'] = (minquote['close']/minquote['open']-1).abs()
    minquote.loc[:,'amihud_10min'] = (minquote.loc[:,'ret_min_abs']/minquote.loc[:,'amount']).rolling(10,min_periods=5).sum()
    return minquote

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class RandomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, train_label):
        self.train_label = train_label
        self.train_label['date'] = self.train_label['time'].dt.date

    def __iter__(self):
        # indices = list(range(len(self.data_source)))
        for Fund_code in self.train_label['Fund_code'].drop_duplicates():
            temp_train_label = self.train_label[self.train_label['Fund_code'] == Fund_code]
            for d in temp_train_label['time'].dt.date.drop_duplicates():
                yield temp_train_label[temp_train_label['time'].dt.date == d].index.tolist()

    def __len__(self):
        return len(self.train_label.drop_duplicates(['date','Fund_code'])) # 返回数据集的总长度

class ALSTM(nn.Module):
    def __init__(self,
                input_size,
                embedding_dim, 
                hidden_size, 
                Q, 
                num_layers = 1,
                beta = 1):
        super(ALSTM, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.Q = Q
        self.num_layers = num_layers
        self.beta = beta
        #self.normalize = nn.LayerNorm(embedding_dim)
        self.MappingLayer = nn.Linear(input_size,embedding_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.LSTMLayer = nn.LSTM(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,Q)
        self.fc2 = nn.Linear(Q, 1, bias=False)
        self.final_map = nn.Linear(hidden_size*2,1)

    def AttentionLayer(self, h_s):
        x = self.tanh(self.fc1(h_s)) 
        x = self.fc2(x)
        weighted_vector = torch.exp(x)/torch.sum(torch.exp(x),dim = 1).unsqueeze(1)  #weight of each time stamp
        a_s = torch.bmm(h_s.transpose(1,2),weighted_vector).squeeze(2)
        return a_s

    def forward(self, input):
        #input = self.normalize(input)
        #input = self.tanh(self.MappingLayer(input))  # 先映射到隐藏层空间
        input = self.leakyrelu(self.MappingLayer(input))
        output, (hn, cn) = self.LSTMLayer(input)
        a_s = self.AttentionLayer(output)
        e_s = torch.cat([a_s,output[:,-1,:]],dim = 1)
        #y = self.tanh(self.final_map(e_s)) #输出为(-1,1)之间的值
        y = self.sigmoid(self.final_map(e_s))
        return y 

def get_adv(origin_input,y_label,final_map,model,epsilon,criterion):
    '''
    origin_input: batch_size * T * feature_dim
    y_label: batch_size
    final_map: final mapping layer
    model: Attentive lSTM
    epsilon: learning rate to control the adv examples
    criterion: loss function
    '''
    e_s = model(origin_input)
    e_s.retain_grad()
    y_s = torch.sign(final_map(e_s)).squeeze(1)
    loss_1 = criterion(y_s,y_label)
    g_s = torch.autograd.grad(outputs = loss_1,inputs=e_s,grad_outputs=None)[0]
    g_snorm = torch.sqrt(torch.norm(g_s,p = 2))
    if g_snorm == 0:
        return 0
    else:
        r_adv = epsilon(g_s/g_snorm)
        return r_adv.detach()
    
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
    def forward(self,y_predict,y_label):
        #y_label,y_predict都是长度为batch_size的一维tensor
        if y_label.shape == y_predict.shape:
            # 这里可能需要加上一个时间权重
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight = torch.arange(y_label.shape[0], 0, -1).unsqueeze(1).float().to(device)
            loss = torch.sum(F.softmax(weight,dim=0)*torch.max(torch.zeros_like(y_label),torch.ones_like(y_label)-y_label * y_predict))
            return loss
        else:
            print(y_label.shape)
            print(y_predict.shape)
            raise Exception("The size of label and predicted value is not equal !")
    
def train(factor_model, dataloader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            y_pred = factor_model(inputs)
            #print(y_pred)
            loss = criterion(y_pred, labels.unsqueeze(1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(factor_model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for inputs, labels  in dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            y_pred = factor_model(inputs)
            loss = criterion(y_pred, labels.unsqueeze(1))
            total_loss += loss.item() 
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == '__main__':
    localpath = 'ETF_T0'
    with open(f'{localpath}/test1.yaml', 'r') as file:
        args = DictToObj(yaml.safe_load(file))
    set_seed(args.SEED)

    folder_name = f"{localpath}/data"
    # 判断文件夹是否存在
    if not os.path.exists(folder_name):
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_name)
        # 这里可能需要计算更多的特征
        ETF_minquotes = pd.read_parquet(f'{localpath}/ETF_minquotes.parquet')
        F588080 = ETF_minquotes.loc[ETF_minquotes['windcode'] == '588080.SH']
        F159915 = ETF_minquotes.loc[ETF_minquotes['windcode'] == '159915.SZ']
        minquotes = {'F588080': F588080, 'F159915': F159915}
        minquotes_Train = {}
        minquotes_Valid = {}
        minquotes_Test = {}
        for fc in ['F588080', 'F159915']:
            minquotes[fc].loc[:,'htc_ret'] = minquotes[fc].groupby(minquotes[fc].index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values
            minquotes[fc] = CalcFinalFactors(minquotes[fc])  #计算因子值
            for factor in args.FACTOR_LIST:
                minquotes[fc][factor] = minquotes[fc][factor].rolling(480,min_periods=240).rank(pct=True)
            minquotes_Train[fc] = minquotes[fc].loc[(minquotes[fc].index.date >= args.TRAIN.START_DATE) & (minquotes[fc].index.date <= args.TRAIN.END_DATE)]
            minquotes_Valid[fc] = minquotes[fc].loc[(minquotes[fc].index.date >= args.VALID.START_DATE) & (minquotes[fc].index.date <= args.VALID.END_DATE)]
            minquotes_Test[fc] = minquotes[fc].loc[(minquotes[fc].index.date >= args.TEST.START_DATE) & (minquotes[fc].index.date <= args.TEST.END_DATE)]

        Train_data, Train_label = make_dataset(minquotes_Train['F588080'], minquotes_Train['F159915'], args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME, seq_gap = args.SEQ_GAP)
        Valid_data, Valid_label = make_dataset(minquotes_Valid['F588080'], minquotes_Valid['F159915'], args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME, seq_gap = args.SEQ_GAP)
        #Test_data, Test_label = 0, 0
        Test_data, Test_label = make_dataset(minquotes_Test['F588080'], minquotes_Test['F159915'], args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME, FillNAN = True) # 需要填充nan值
        for data_name, data in zip(['Train_data', 'Train_label', 'Valid_data', 'Valid_label', 'Test_data', 'Test_label'], [Train_data, Train_label, Valid_data, Valid_label, Test_data, Test_label]):
            # if data_name in ['Test_data','Test_label']:  #暂时先不保存这里的Test
            #     continue
            with open(f'{localpath}/data/{data_name}.pickle', 'wb') as file:
                pickle.dump(data, file)

    else:
        AllData = []
        for data_name in ['Train_data', 'Train_label', 'Valid_data', 'Valid_label', 'Test_data', 'Test_label']:
            # if data_name in ['Test_data','Test_label']:
            #     continue
            with open(f'{localpath}/data/{data_name}.pickle', 'rb') as file:
                loaded_data = pickle.load(file)
            AllData.append(loaded_data)
        #Train_data, Train_label, Valid_data, Valid_label = tuple(AllData)
        Train_data, Train_label, Valid_data, Valid_label, Test_data, Test_label = tuple(AllData)
    
    #TrainBatchSampler = RandomBatchSampler(Train_label)
    #ValidBatchSampler = RandomBatchSampler(Valid_label)
    #TrainDataloader = DataLoader(MyDataset(Train_data, Train_label['htc_ret'].values), batch_sampler=TrainBatchSampler, pin_memory=True)
    #ValidDataloader = DataLoader(MyDataset(Valid_data, Valid_label['htc_ret'].values), batch_sampler=ValidBatchSampler, pin_memory=True)
    #TestDataloader = DataLoader(MyDataset(Test_data, Test_label['htc_ret'].values), pin_memory=True)
    TrainDataloader = DataLoader(MyDataset(Train_data, Train_label['htc_ret'].values), batch_size=args.BATCH_SIZE, pin_memory=True)
    ValidDataloader = DataLoader(MyDataset(Valid_data, Valid_label['htc_ret'].values), batch_size=args.BATCH_SIZE, pin_memory=True)
    print('数据准备完毕')  # 缺少数据归一化的操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALSTM(input_size=len(args.FACTOR_LIST),
                embedding_dim = args.EMBEDDING_DIM, 
                hidden_size=args.HIDDEN_SIZE, 
                Q = args.Q, 
                num_layers=args.NUM_LAYERS)
    model.to(device)
    #criterion = HingeLoss()
    #criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()

    #L2 normalization
    # weight_list,bias_list = [],[]
    # for name,p in model.named_parameters():
    #     if 'bias' in name:
    #         bias_list += [p]
    #     else:
    #         weight_list += [p]
    # optimizer = torch.optim.SGD([{'params': weight_list, 'weight_decay':1e-5},
    #                     {'params': bias_list, 'weight_decay':0}],
    #                     lr = args.LEARNING_RATE,
    #                     momentum = 0.9)

    #初始化：Xavier Initialization
    for n in model.modules():  
        if isinstance(n,nn.Linear): #线性全连接层初始化 
            n.weight = nn.init.xavier_normal_(n.weight, gain=1.) 

    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    # Start Trainig
    early_stopping_counter = 0 
    for epoch in tqdm(range(args.NUM_EPOCHS)):
        train_loss = train(model, TrainDataloader, criterion, optimizer)
        val_loss = validate(model, ValidDataloader, criterion)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{localpath}/model.pth')
            print(f"Model saved")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= args.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break