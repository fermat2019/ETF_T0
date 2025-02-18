from utils import *
import argparse
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

def make_dataset(Fund_data, seq_len, factor_list, label_name, FillNAN = False, seq_gap = 1):
    ## 这里可能需要考虑填充nan值
    data = []
    label = []
    Fund_data = Fund_data[Fund_data.index.strftime('%H:%M')<='14:49']
    Fund_data.loc[:,'time'] = Fund_data.groupby(Fund_data.index.date)['close'].cumcount().astype(float)+1
    Fund_data.loc[:,'time'] = Fund_data.groupby(Fund_data.index.date)['time'].transform(lambda x: x/x.sum())
    #Fund_data[label_name] = Fund_data[label_name].apply(lambda x: 0 if x<=0 else(1 if x>0 else np.nan))  #用于计算分类熵
    for d in tqdm(np.unique(Fund_data.index.date)):
        Fund_temp = Fund_data.loc[Fund_data.index.date == d].sort_index()
        for i in range(seq_len, len(Fund_temp), seq_gap):
            ## 考虑每个特征时序相对位置的变化，即归一化操作
            #temp_data = Fund_temp.iloc[i-seq_len:i][factor_list].rank(axis=0,pct=True).values
            temp_data = Fund_temp.iloc[i-seq_len:i][factor_list].values  #不做归一化，每个因子都是过去一段时间的分位数
            temp_label = Fund_temp.iloc[i-1][label_name]  #判断大于0还是小于0
            
            if FillNAN:
                temp_data[np.isnan(temp_data)] = 0.5   #用排序中间值0.5填充nan值
                data.append(temp_data) 
                label.append((Fund_temp.iloc[i-1].name,temp_label))          
            elif np.isnan(temp_data).sum()+np.isnan(temp_label) == 0:
                data.append(temp_data)
                label.append((Fund_temp.iloc[i-1].name,temp_label))
            else:
                continue
    return np.array(data), pd.DataFrame(label,columns=['time','htc_ret'])


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class RandomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, train_label, batch_size):
        self.train_label = train_label
        self.train_label['date'] = self.train_label['time'].dt.date
        self.batch_size = batch_size

    def __iter__(self):
        # for d in temp_train_label['time'].dt.date.drop_duplicates():  #每次yield同一天的数据
        #     yield temp_train_label[temp_train_label['time'].dt.date == d].index.tolist()
        for HM in self.train_label['time'].dt.strftime('%H:%M').drop_duplicates():
            OutputList = self.train_label[self.train_label['time'].dt.strftime('%H:%M') == HM].index.tolist()
            for i in range(0,len(OutputList),self.batch_size):
                yield OutputList[i:i+self.batch_size]

    def __len__(self):
        time_len = len(self.train_label['time'].dt.strftime('%H:%M').drop_duplicates())
        return time_len*((len(self.train_label['time'].dt.date.drop_duplicates())-1)//self.batch_size+1) # 返回数据集的总长度

class ALSTM(nn.Module):
    def __init__(self,
                input_size,
                embedding_dim, 
                hidden_size, 
                query_size, 
                num_layers = 1,
                beta = 1):
        super(ALSTM, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.num_layers = num_layers
        self.beta = beta
        #self.normalize = nn.LayerNorm(embedding_dim)
        self.MappingLayer = nn.Linear(input_size,embedding_dim)
        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.LSTMLayer = nn.LSTM(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,query_size)
        self.fc2 = nn.Linear(query_size, 1, bias=False)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout 层
        #self.final_map = nn.Linear(hidden_size*2,1)

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
        input = self.dropout1(input)
        output, (hn, cn) = self.LSTMLayer(input)
        output = self.dropout1(output)
        a_s = self.AttentionLayer(output)
        e_s = torch.cat([a_s,output[:,-1,:]],dim = 1)
        #y = self.tanh(self.final_map(e_s)) #输出为(-1,1)之间的值
        #y = self.final_map(e_s)
        return e_s

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
    y_s = final_map(e_s)
    loss_1 = criterion(y_s,y_label)
    g_s = torch.autograd.grad(outputs = loss_1,inputs=e_s,grad_outputs=None)[0]
    g_snorm = torch.sqrt(torch.norm(g_s,p = 2))
    if g_snorm == 0:
        return 0
    else:
        r_adv = epsilon*(g_s/g_snorm)
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
    
class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss,self).__init__()
    def forward(self,y_predict,y_label):
        #y_label,y_predict都是长度为batch_size的一维tensor
        y_label_mean = torch.mean(y_label)
        y_predict_mean = torch.mean(y_predict)
        loss = -torch.mean((y_label-y_label_mean)*(y_predict-y_predict_mean))/(torch.std(y_label)*torch.std(y_predict))
        return loss

def train(factor_model, final_map, dataloader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            e_s = factor_model(inputs)
            #print(y_pred)
            r_adv = get_adv(inputs,labels.unsqueeze(1),final_map,factor_model,1e-2,criterion)
            loss1 = criterion(final_map(e_s), labels.unsqueeze(1))
            loss2 = criterion(final_map(e_s+r_adv), labels.unsqueeze(1))
            loss = (loss1+loss2)/2
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(factor_model, final_map, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for inputs, labels  in dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            e_s = factor_model(inputs)
            y_pred = final_map(e_s)
            loss = criterion(y_pred, labels.unsqueeze(1))
            total_loss += loss.item() 
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == '__main__':
    localpath = '.'
    parser = argparse.ArgumentParser(description='Choose a YAML configuration file.')
    parser.add_argument('YAML_file', type=str, choices=['F588080', 'F159915'])
    parser.add_argument('Train_year', type=int, choices=[2022, 2023])
    parserargs = parser.parse_args()
    FundCode = parserargs.YAML_file
    TrainYear = parserargs.Train_year

    with open(f'{localpath}/configs/train.yaml', 'r') as file:
        args = DictToObj(yaml.safe_load(file))
    set_seed(args.SEED)
    model_path = f"{localpath}/models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    folder_name = f"{localpath}/data"
    if not os.path.exists(folder_name):
        # 如果文件夹不存在，则创建它，并构造数据
        os.makedirs(folder_name)
        ETF_minquotes = pd.read_parquet(f'{localpath}/ETF_minquotes.parquet')
        F588080 = ETF_minquotes.loc[ETF_minquotes['windcode'] == '588080.SH']
        F159915 = ETF_minquotes.loc[ETF_minquotes['windcode'] == '159915.SZ']
        minquotes = {'F588080': F588080, 'F159915': F159915}
        factor_list = args.FACTOR_LIST + ['time']


        
        for fc in minquotes.keys():
            print(f'Processing {fc}...')
            minquotes[fc].loc[:,'htc_ret'] = minquotes[fc].groupby(minquotes[fc].index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values
            minquotes[fc] = CalcFinalFactors(minquotes[fc])  #计算因子值
            for factor in args.FACTOR_LIST:
                minquotes[fc][factor] = minquotes[fc][factor].rolling(args.ROLLING_WINDOW, min_periods=240).rank(pct=True)

            iterator = zip([('Train_data_2022','Train_label_2022'),('Train_data_2023','Train_label_2023'),
                ('Valid_data_2022','Valid_label_2022'),('Valid_data_2023','Valid_label_2023'),
                ('Test_data','Test_label')],
                [(args.TRAIN2022.START_DATE,args.TRAIN2022.END_DATE),(args.TRAIN2023.START_DATE,args.TRAIN2023.END_DATE),
                (args.VALID2022.START_DATE,args.VALID2022.END_DATE),(args.VALID2023.START_DATE,args.VALID2023.END_DATE),
                (args.TEST.START_DATE,args.TEST.END_DATE)])
            
            for (data_name, label_name), (startdate, enddate) in iterator:
                temp_minquotes = minquotes[fc].loc[(minquotes[fc].index.date >= startdate) & \
                                                   (minquotes[fc].index.date <= enddate)]
                fillna = True if data_name == 'Test_data' else False
                temp_data, temp_label = make_dataset(temp_minquotes, args.SEQ_LEN, factor_list, args.LABEL_NAME, seq_gap = args.SEQ_GAP, FillNAN=fillna)
                with open(f'{localpath}/data/{data_name}_{fc}.pickle', 'wb') as file:
                    pickle.dump(temp_data, file)
                with open(f'{localpath}/data/{label_name}_{fc}.pickle', 'wb') as file:
                    pickle.dump(temp_label, file)

    AllData = []
    for data_name in ['Train_data', 'Train_label', 'Valid_data', 'Valid_label']:
        with open(f'{localpath}/data/{data_name}_{TrainYear}_{FundCode}.pickle', 'rb') as file:
            loaded_data = pickle.load(file)
        AllData.append(loaded_data)
    Train_data, Train_label, Valid_data, Valid_label = tuple(AllData)
    
    TrainBatchSampler = RandomBatchSampler(Train_label,args.BATCH_SIZE)
    ValidBatchSampler = RandomBatchSampler(Valid_label,args.BATCH_SIZE)
    TrainDataloader = DataLoader(MyDataset(Train_data, Train_label['htc_ret'].values), batch_sampler=TrainBatchSampler, pin_memory=True)
    ValidDataloader = DataLoader(MyDataset(Valid_data, Valid_label['htc_ret'].values), batch_sampler=ValidBatchSampler, pin_memory=True)
    #TestDataloader = DataLoader(MyDataset(Test_data, Test_label['htc_ret'].values), pin_memory=True)
    #TrainDataloader = DataLoader(MyDataset(Train_data, Train_label['htc_ret'].values), batch_size=args.BATCH_SIZE, pin_memory=True)
    #ValidDataloader = DataLoader(MyDataset(Valid_data, Valid_label['htc_ret'].values), batch_size=args.BATCH_SIZE, pin_memory=True)
    print('数据准备完毕')  # 缺少数据归一化的操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALSTM(input_size=len(args.FACTOR_LIST)+1,
                embedding_dim = args.EMBEDDING_DIM, 
                hidden_size=args.HIDDEN_SIZE, 
                query_size = args.QUERY_SIZE, 
                num_layers=args.NUM_LAYERS)
    final_map = nn.Sequential(
    nn.Linear(2 * args.HIDDEN_SIZE, 1, bias=True),
    nn.Sigmoid()
    )
    #final_map = nn.Linear(2*args.HIDDEN_SIZE,1,bias = True)
    model.to(device)
    final_map.to(device)
    #criterion = HingeLoss()
    #criterion = nn.MSELoss()
    criterion = ICLoss()
    #criterion = nn.BCELoss()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)

    # Start Trainig
    early_stopping_counter = 0 
    for epoch in tqdm(range(args.NUM_EPOCHS)):
        train_loss = train(model, final_map, TrainDataloader, criterion, optimizer)
        val_loss = validate(model, final_map, ValidDataloader, criterion)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            check_point = {'model':model.state_dict(), 'final_map':final_map.state_dict()}
            torch.save(check_point, f'{localpath}/models/model_{TrainYear}_{FundCode}.pth')
            print(f"Model saved")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= args.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break