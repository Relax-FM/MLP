import pandas as pd
from torch.utils.data import Dataset

def dataload_xlsx(file):
    '''
    Use for loading data about candles from file
    :arg:
    file - name of file to loading data (file.xls)
    :return: pd.DateFrame about candle in format (Date, Open, High, Low, Close, EMA200, Assets, Take_profit, Stop_loss)
    '''
    data = pd.read_excel(f'{file}.xlsx')
    data = data.set_index('Date')
    del data['Close']
    del data['Open']
    # print(data)
    return data

def dataload_csv(file):
    '''
    Use for loading data about candles from file
    :arg:
    file - name of file to loading data (file.csv)
    :return: pd.DateFrame about candle in format (Date, Open, High, Low, Close, EMA200, Assets, Take_profit, Stop_loss)
    '''
    data = pd.read_csv(f'{file}.csv')
    return data

class PreDataLoader(Dataset):
    """
    Class for correct load all data
    """
    def __init__(self, data, pred_size=4, label_size=2, candle_count=50, start=200, stop=350, normalization_pred=True, normalization_label=False, label_offset=0):
        """Create DataLoader for your DataSet
            :Parameters:
                tickers : str, list
                    List of tickers to download
                data: DataFrame
                    DataFrame of candle parameters
                pred_size: int
                    Prediction size(Count of columns with prediction parameters)
                label_size: int
                    Label size(Count of columns with label parameters)
                butch_size: int
                    Butch size for NN
                start: int
                    Number of row that you want to start
                stop: int
                    Number of row that you want to finish
            """
        self.data = data.copy()
        self.data2 = data.copy()
        self.label_offset = label_offset
        self.pred_size = pred_size
        self.label_size = label_size
        self.candle_count = candle_count
        self.start = start
        self.stop = stop
        self.predictions = []
        self.labels = []
        self.butches = []
        self.normalization_pred = normalization_pred
        # if (normalization_pred == True) and (normalization_label==False):
        #     self.minmax_normalization(flag=0)
        # elif (normalization_label == True) and (normalization_pred == False):
        #     self.minmax_normalization(flag=1)
        # elif (normalization_label == True) and (normalization_pred == True):
        #     self.minmax_normalization(flag=2)
        self.create_exit_data()

    def minmax_normalization(self, flag=0):
        # self.data - тут храниться вся инфа в формате DataType
        # Её нужно нормализовать со строки self.start до строки self.stop
        # Нормализовать по первым 3 столбцам (high, low, ema200) тоесть столбцы от 0 до self.pred_size-1
        # Ну и для дебага вывести в формате xls
        for i in range(self.start, self.stop-self.candle_count):
            min0 = self.data2.iloc[:,1:2][i:(i+self.candle_count)].min().iloc[0]
            max0 = self.data2.iloc[:,0:1][i:(i+self.candle_count)].max().iloc[0]

            if (flag==0) or (flag==2):
              for j in range(self.pred_size-1):
                x = self.data2.iloc[:,j][i+self.candle_count-1]
                self.data.iloc[:,j][i+self.candle_count-1]=(x-min0)/(max0-min0)
            if (flag==1) or (flag==2):
              for j in range(self.pred_size+self.label_offset, self.pred_size+self.label_size+self.label_offset):
                x = self.data2.iloc[:,j][i+self.candle_count-1]
                self.data.iloc[:,j][i+self.candle_count-1]=(x-min0)/(max0-min0)
        self.data.to_excel('data1.xlsx')


    def create_exit_data(self):
        '''Not availaible now'''
        prediction = []
        label = [] if self.label_size>1 else 0
        # if (self.label_size>1):
        #     label=[]
        # elif (self.label_size == 1):
        #     label = 0
        # else:
        #     raise Exception('Нет информации о лейбах')
        for i in range(self.start, self.stop-self.candle_count):
            for j in range(self.pred_size-1):
                for k in range(self.candle_count):
                    prediction.append(self.data.iloc[:, j][i+k])
            prediction.append(self.data.iloc[:, self.pred_size-1][i+self.candle_count-1])
            for j in range(self.label_size):
                if (self.label_size == 1):
                    label = self.data.iloc[:, j+self.pred_size+self.label_offset][i+self.candle_count-1]
                else:
                    label.append(self.data.iloc[:, j+self.pred_size+self.label_offset][i+self.candle_count-1])

            if(self.normalization_pred == True):
                max0 = max(prediction[0:2*self.candle_count])
                min0 = min(prediction[0:2*self.candle_count])
                for i in range(len(prediction)):
                    prediction[i] = (prediction[i]-min0)/(max0-min0)

            self.predictions.append(prediction)
            self.labels.append(label)
            label = []
            prediction = []

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        self.i = self.i+1
        if self.i < (self.stop-self.candle_count)-self.start-1:
            return self.predictions[self.i], self.labels[self.i]
        else:
            raise StopIteration

    def get_data(self):
        exit_data = [[self.predictions[i], self.labels[i]] for i in range(len(self.labels))]
        return exit_data