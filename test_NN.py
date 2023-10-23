import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
from load_data import dataload_csv, dataload_xlsx, PreDataLoader
from Dataset import NASDAQDataSet
from MLP import NN_Nasdaq
import yaml
from labels import get_label
import visualisation_func as vf


if __name__ == '__main__':

    options_path = 'config.yml'
    with open(options_path, 'r') as options_stream:
        options = yaml.safe_load(options_stream)

    network_options = options.get('network')
    dataset_options = options.get('dataset')
    dataset_sizes = dataset_options.get('sizes')

    device = network_options['device']  # 'cpu'
    label_offset = get_label(dataset_options['label_name'])
    batch_size = dataset_options['batch_size']  # Кол-во элементов в баче
    candle_count = dataset_options['candle_count']  # Кол-во отсматриваемых свечей в баче
    hidden_layer_size = network_options['hidden_layer_size']  # Размер скрытого слоя
    output_layer_size = network_options['output_layer_size']  # размер выходного слоя
    correct_label_size = dataset_options['correct_label_size']  # Кол-во столбцов с ответами в датасете
    start_position = dataset_sizes['test_start_position']  # Начальная позиция датасета (как бы с 200 позиции но по факту будет создавать для start_position+candle_count)
    stop_position = dataset_sizes['test_stop_position']  # Конечная позиция датасета (Правда конечная позиция. Создает датасет до stop_position позиции )

    candles_params_count = dataset_options['candles_params_count']  # Кол-во столбцов с параметрами свечи
    additional_params_count = dataset_options['additional_params_count']  # Дополнительный столбец с параметрами свечи
    info_label_size = candles_params_count + additional_params_count  # Кол-во столбцов с параметрами свечи

    model = NN_Nasdaq(input_size=candle_count * candles_params_count + additional_params_count, hidden_size=hidden_layer_size, output_size=output_layer_size)

    path_name = 'model_take_profit_' + device + '.pth' if label_offset == 0 else 'model_stop_loss_' + device + '.pth'

    print(f'Start with model {path_name}')

    model.load_state_dict(torch.load(path_name))
    ''' Вот здесь менять model.pth на model1.pth, чтобы перейти от take-profit к stop-loss'''

    dataset_MSFT = dataload_xlsx(dataset_options['file_name'])  # Грузим датасет из файла 'test'

    DL = PreDataLoader(data=dataset_MSFT, pred_size=info_label_size, label_size=correct_label_size,
                       candle_count=candle_count, start=start_position, stop=stop_position,
                       normalization_pred=dataset_options['normalization_pred'])
    batches = DL.get_data()
    dataset = NASDAQDataSet(batches)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=0
    )

    labels = []
    results = []

    for info, label in (pbar := tqdm(dataloader)):
        print(label)
        labels.append(label.detach().numpy()[0])
        result = model(info)
        results.append(result.detach().numpy()[0])
        print(result)
        relation = (result/label)*100
        print(f'Relation is : {relation}%')
        print('#'*30)

    print(labels)
    print(results)

    avg_lbl, avg_res = vf.average(labels, results)
    standard_deviation = vf.calculated_standard_deviation(labels, results)
    error = vf.calculated_error(standard_deviation, avg_lbl)
    max_error = vf.calculated_max_error(labels, results)

