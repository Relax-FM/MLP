import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
from load_data import dataload_xlsx, PreDataLoader
from Dataset import NASDAQDataSet
from torch.cuda.amp import autocast, GradScaler
from optimizer import get_optimizer
from losses import get_losser
from labels import get_label
import visualisation_func as vf

class NN_Nasdaq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size) # входной слой
        self.linear2 = nn.Linear(hidden_size, hidden_size) # Можно пихнуть еще один скрытый слой для тестов
        self.linear3 = nn.Linear(hidden_size, output_size) # скрытый слой

        self.act_func = nn.ReLU() # Задали функцию активации
        self.flat = nn.Flatten()
        '''self.model = nn.Sequential(
            linear1,
            act_func,
            linear2
        )'''

    def forward(self, x):
        #print(x)
        out = self.flat(x) # возможно это можно закоментить
        out = self.linear1(out)
        out = self.act_func(out)
        out = self.linear2(out)
        out = self.act_func(out)
        out = self.linear3(out)
        out = out.reshape((out.shape[0]))
        return out

def accuracy(pred, label, epsilon=1):
    count = 0
    predict = pred.detach().numpy()
    lbl = label.detach().numpy()
    for i in range(len(label)):
        accuracy_now = (lbl[i]-predict[i])*100
        if (accuracy_now<epsilon):
            count+=1
    return count

def lossing(pred,label):
    count = 0
    predict = pred.detach().numpy()
    lbl = label.detach().numpy()
    for i in range(len(label)):
        loss_now = np.abs(lbl[i]-predict[i])
        count += loss_now
    return count

if __name__ == '__main__':

    """
    Date,Open,High,Low,Close,EMA200,Assets,Take_profit,Stop_loss
    """

    options_path = 'config.yml'
    with open(options_path, 'r') as options_stream:
        options = yaml.safe_load(options_stream)

    network_options = options.get('network')
    dataset_options = options.get('dataset')
    dataset_sizes = dataset_options.get('sizes')
    optimizer_options = network_options.get('optimizer')
    losser_options = network_options.get('loss')

    device = network_options['device']  # 'cpu'
    label_offset = get_label(dataset_options['label_name'])
    batch_size = dataset_options['batch_size']  # Кол-во элементов в баче
    candle_count = dataset_options['candle_count']  # Кол-во отсматриваемых свечей в баче
    hidden_layer_size = network_options['hidden_layer_size']  # Размер скрытого слоя
    output_layer_size = network_options['output_layer_size']  # размер выходного слоя
    correct_label_size = dataset_options['correct_label_size']  # Кол-во столбцов с ответами в датасете
    start_position = dataset_sizes['start_position']  # Начальная позиция датасета (как бы с 200 позиции но по факту будет создавать для start_position+candle_count)
    stop_position = dataset_sizes['stop_position']  # Конечная позиция датасета (Правда конечная позиция. Создает датасет до stop_position позиции )




    candles_params_count = dataset_options['candles_params_count'] # Кол-во столбцов с параметрами свечи
    additional_params_count = dataset_options['additional_params_count'] # Дополнительный столбец с параметрами свечи
    info_label_size = candles_params_count + additional_params_count # Кол-во столбцов с параметрами свечи

    dataset_MSFT = dataload_xlsx(dataset_options['file_name'])  # Грузим датасет из файла 'test'

    DL = PreDataLoader(data=dataset_MSFT, pred_size=info_label_size, label_size=correct_label_size,
                       candle_count=candle_count, start=start_position, stop=stop_position, label_offset=label_offset,
                       normalization_pred=dataset_options['normalization_pred'])
    batches = DL.get_data()
    #print(batches)
    dataset = NASDAQDataSet(batches)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=dataset_options['shuffle'], num_workers=dataset_options['num_workers']
    )

    model = NN_Nasdaq(input_size=candle_count * candles_params_count + additional_params_count, hidden_size=hidden_layer_size, output_size=output_layer_size)

    loss_func = get_losser(losser_options)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = get_optimizer(model.parameters(), optimizer_options)

    model = model.to(device)
    loss_func = loss_func.to(device)

    use_amp = network_options['use_amp']
    #scaler = torch.cuda.amp.GradScaler()

    epochs = network_options['epochs']
    count = 0
    losses = []
    accuracies = []
    labels = []
    results = []
    start_time = time.time()

    for epoch in range(epochs):
        count += 1
        loss_val = 0
        acc_val = 0
        loss_my = 0
        for info, label in dataloader:
            optimizer.zero_grad()  # Обнуляем градиенты, чтобы они не помешали нам на прогоне новой картинки

            info = info.to(device)
            label = label.to(device)

            # label = F.one_hot(label,10).float()

            with autocast(use_amp, dtype = torch.float16):
                pred = model(info)
                loss = loss_func(pred, label)  # посчитали ошибку (значение в label - значение полученое нашим model(img))

            loss.backward()  # Прошелись по всему графу вычислений и посчитали все градики для нейронов

            loss_item = loss.item()
            loss_val += loss_item
            loss_my_item = lossing(pred.cpu(), label.cpu())
            loss_my += loss_my_item

            optimizer.step()  # Сделали шаг градиентным спуском с учетом градиента посчитанного loss.backward()

            acc_current = accuracy(pred.cpu(), label.cpu(), epsilon=0.1)
            acc_val += acc_current

            labels.append(label.cpu())
            results.append(pred.cpu())

        # смотрим какая ошибка на одной картинке loss_item
        print(f'epoch: {count}\tloss: {loss_my / 110}\tacuraccy: {acc_val / 110}')
        accuracies.append(acc_val / 110)
        losses.append(loss_my / 110)
    print(f'Full time learning : {time.time() - start_time}')

    print(labels)
    print(results)
    labels_np = labels.numpy()
    results_np = results.numpy()
    avg_lbl, avg_res = vf.average(labels_np, results_np)
    standard_deviation = vf.calculated_standard_deviation(labels_np, results_np)
    error = vf.calculated_error(standard_deviation, avg_lbl)
    max_error = vf.calculated_max_error(labels_np, results_np)

    path_name = 'model_take_profit_'+device+'.pth' if label_offset == 0 else 'model_stop_loss_'+device+'.pth'
    print(f'Save model as {path_name}')
    torch.save(model.state_dict(), path_name)

    h = np.linspace(1, len(losses), len(losses))

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(h[:], losses[:])
    ax.set_title("Loss for epoch.")
    ax.set_xlabel("Axis epoch")
    ax.set_ylabel("Axis loss")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(h, accuracies)
    ax.set_title("accuracy for epoch.")
    ax.set_xlabel("Axis epoch")
    ax.set_ylabel("Axis accuracy")
    ax.grid()
    plt.show()