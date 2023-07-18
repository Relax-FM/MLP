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
    def __init__(self, data, pred_size=4, label_size=2, candle_count=50, start=200, stop=350):
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
        self.data = data
        self.pred_size = pred_size
        self.label_size = label_size
        self.candle_count = candle_count
        self.start = start
        self.stop = stop
        self.predictions = []
        self.labels = []
        self.butches = []

    def create_exit_data(self):
        prediction = []
        label = []
        for i in range(self.start, self.stop-self.candle_count):
            for j in range(self.pred_size-1):
                for k in range(self.candle_count):
                    prediction.append(self.data.iloc[:, j][i+k])
            prediction.append(self.data.iloc[:, self.pred_size-1][i+self.candle_count-1])
            for j in range(self.label_size):
                label.append(self.data.iloc[:, j+self.pred_size][i+self.candle_count-1])
            self.predictions.append(prediction)
            self.labels.append(label)
            label = []
            prediction = []

    def __iter__(self):
        self.i=-1
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