import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
from load_data import dataload_csv, dataload_xlsx, PreDataLoader
from Dataset import NSDAQDataSet
from MLP import NN_Nasdaq

if __name__ == '__main__':

    batch_size = 25  # Кол-во бачей
    candle_count = 50  # Кол-во отсматриваемых свечей в баче
    hidden_layer_size = 25  # Размер скрытого слоя
    output_layer_size = 1  # размер выходного слоя
    info_label_size = 4  # Кол-во столбцов с параметрами свечи
    correct_label_size = 1  # Кол-во столбцов с ответами в датасете
    start_position = 300  # Начальная позиция датасета
    stop_position = 362  # Конечная позиция датасета

    candles_params_count = 3  # Кол-во столбцов с параметрами свечи
    additional_params_count = 1  # Дополнительный столбец с параметрами свечи

    model = NN_Nasdaq(input_size=candle_count * candles_params_count + additional_params_count, hidden_size=hidden_layer_size, output_size=output_layer_size)
    model.load_state_dict(torch.load('model.pth'))
    ''' Вот здесь менять model.pth на model1.pth, чтобы перейти от take-profit к stop-loss'''

    dataset_MSFT = dataload_xlsx('test')  # Грузим датасет из файла 'test'

    DL = PreDataLoader(data=dataset_MSFT, pred_size=info_label_size, label_size=correct_label_size,
                       candle_count=candle_count, start=start_position, stop=stop_position, normalization_label=False,
                       normalization_pred=True)
    batches = DL.get_data()
    dataset = NSDAQDataSet(batches)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=1
    )

    for info, label in (pbar := tqdm(dataloader)):
        print(label)
        print(model(info))
        print('#'*30)