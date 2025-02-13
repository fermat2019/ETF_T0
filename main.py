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

def make_dataset(F588080_temp, F159915_temp, seq_len, factor_list, label_name, seq_gap = 1):
    data = []
    label = []
    for Fund_code, Fund_temp in zip(['F588080.SH', 'F159915.SZ'],[F588080_temp, F159915_temp]):
        Fund_temp = Fund_temp[Fund_temp.index.strftime('%H:%M')<='14:00']
        for i in tqdm(range(seq_len, len(Fund_temp), seq_gap)):
            ## 考虑每个特征时序相对位置的变化，即归一化操作
            temp_data = Fund_temp.iloc[i-seq_len:i][factor_list].rank(axis=0,pct=True).values
            temp_label = Fund_temp.iloc[i-1][label_name]
            # index = Fund_temp.iloc[i-1].index
            
            if np.isnan(temp_data).sum()+np.isnan(temp_label) == 0:
                data.append(temp_data)
                label.append((Fund_temp.iloc[i-1].name,Fund_code,temp_label))
    return np.array(data), pd.DataFrame(label,columns=['time','Fund_code','htc_ret'])


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
            for d in self.train_label['time'].dt.date.drop_duplicates():
                yield self.train_label[(self.train_label['Fund_code'] == Fund_code) & (self.train_label['time'].dt.date == d)].index.tolist()

    def __len__(self):
        return len(self.train_label.drop_duplicates(['date','Fund_code'])) # 返回数据集的总长度

class GRUModel(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_layers = 1,
        output_size = 1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, input, y):
        output, _ = self.gru(input)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        loss = F.mse_loss(output, y)  # 损失函数定义在模型当中
        return loss, output
    
def train(factor_model, dataloader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            loss, _ = factor_model(inputs, labels.unsqueeze(1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(factor_model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for inputs, labels  in dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            loss, _ = factor_model(inputs, labels.unsqueeze(1))
            total_loss += loss.item() 
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
        F588080.loc[:,'htc_ret'] = F588080.groupby(F588080.index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values
        F159915.loc[:,'htc_ret'] = F159915.groupby(F159915.index.date).apply(lambda x:x['close'].iloc[-1]/x['open'].shift(-1)-1).values

        F588080_Train = F588080.loc[(F588080.index.date >= args.TRAIN.START_DATE) & (F588080.index.date <= args.TRAIN.END_DATE)]
        F159915_Train = F159915.loc[(F159915.index.date >= args.TRAIN.START_DATE) & (F159915.index.date <= args.TRAIN.END_DATE)]
        F588080_Valid = F588080.loc[(F588080.index.date >= args.VALID.START_DATE) & (F588080.index.date <= args.VALID.END_DATE)]
        F159915_Valid = F159915.loc[(F159915.index.date >= args.VALID.START_DATE) & (F159915.index.date <= args.VALID.END_DATE)]
        F588080_Test = F588080.loc[(F588080.index.date >= args.TEST.START_DATE) & (F588080.index.date <= args.TEST.END_DATE)]
        F159915_Test = F159915.loc[(F159915.index.date >= args.TEST.START_DATE) & (F159915.index.date <= args.TEST.END_DATE)]

        Train_data, Train_label = make_dataset(F588080_Train, F159915_Train, args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME, args.SEQ_GAP)
        Valid_data, Valid_label = make_dataset(F588080_Valid, F159915_Valid, args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME, args.SEQ_GAP)
        Test_data, Test_label = make_dataset(F588080_Test, F159915_Test, args.SEQ_LEN, args.FACTOR_LIST, args.LABEL_NAME)
        for data_name, data in zip(['Train_data', 'Train_label', 'Valid_data', 'Valid_label', 'Test_data', 'Test_label'], [Train_data, Train_label, Valid_data, Valid_label, Test_data, Test_label]):
            with open(f'{localpath}/data/{data_name}.pickle', 'wb') as file:
                pickle.dump(data, file)

    else:
        AllData = []
        for data_name in ['Train_data', 'Train_label', 'Valid_data', 'Valid_label', 'Test_data', 'Test_label']:
            with open(f'{localpath}/data/{data_name}.pickle', 'rb') as file:
                loaded_data = pickle.load(file)
            AllData.append(loaded_data)
        Train_data, Train_label, Valid_data, Valid_label, Test_data, Test_label = tuple(AllData)
    
    TrainBatchSampler = RandomBatchSampler(Train_label)
    ValidBatchSampler = RandomBatchSampler(Valid_label)
    TrainDataloader = DataLoader(MyDataset(Train_data, Train_label['htc_ret'].values), batch_sampler=TrainBatchSampler, pin_memory=True)
    ValidDataloader = DataLoader(MyDataset(Valid_data, Valid_label['htc_ret'].values), batch_sampler=ValidBatchSampler, pin_memory=True)
    TestDataloader = DataLoader(MyDataset(Test_data, Test_label['htc_ret'].values), pin_memory=True)

    print('数据准备完毕')  # 缺少数据归一化的操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size=len(args.FACTOR_LIST), hidden_size=args.HIDDEN_SIZE, num_layers=args.NUM_LAYERS, output_size=1)
    model.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    # Start Trainig
    for epoch in tqdm(range(args.NUM_EPOCHS)):
        train_loss = train(model, TrainDataloader, optimizer)
        val_loss = validate(model, ValidDataloader)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{localpath}/model.pth')
            print(f"Model saved")
