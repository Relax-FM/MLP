import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from load_data import dataload_xlsx, PreDataLoader
from Dataset import NSDAQDataSet

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
        accuracy_now = ((lbl[i]-predict[i])/predict[i])*100
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

    # TODO: Отдебажить с мнистом (сравнить).
    # TODO: 1) - Дебагинг НС в python.

    device = 'cpu'  # 'cpu'
    batch_size = 25  # Кол-во элементов в баче
    candle_count = 50  # Кол-во отсматриваемых свечей в баче
    hidden_layer_size = 25  # Размер скрытого слоя
    output_layer_size = 1  # размер выходного слоя
    info_label_size = 4  # Кол-во столбцов с параметрами свечи
    correct_label_size = 1  # Кол-во столбцов с ответами в датасете
    start_position = 200  # Начальная позиция датасета (как бы с 200 позиции но по факту будет создавать для start_position+candle_count)
    stop_position = 350  # Конечная позиция датасета (Правда конечная позиция. Создает датасет до stop_position позиции )
    label_offset = 1 # 0 - take-profit 1 - stop-loss
    ''' 0 - take-profit 1 - stop-loss '''


    candles_params_count = 3 # Кол-во столбцов с параметрами свечи
    additional_params_count = 1 # Дополнительный столбец с параметрами свечи

    dataset_MSFT = dataload_xlsx('test')  # Грузим датасет из файла 'test'

    DL = PreDataLoader(data=dataset_MSFT, pred_size=info_label_size, label_size=correct_label_size,
                       candle_count=candle_count, start=start_position, stop=stop_position, label_offset=label_offset)
    batches = DL.get_data()
    #print(batches)
    dataset = NSDAQDataSet(batches)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    model = NN_Nasdaq(input_size=candle_count * candles_params_count + additional_params_count, hidden_size=hidden_layer_size, output_size=output_layer_size)

    loss_func = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(device)
    loss_func = loss_func.to(device)

    use_amp = True

    epochs = 1500
    count = 0
    losses = []
    accuracies = []
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
            pred = model(info)

            # print(pred)
            # print(label)
            # print(img)

            loss = loss_func(pred, label)  # посчитали ошибку (значение в label - значение полученое нашим model(img))

            loss.backward()  # Прошелись по всему графу вычислений и посчитали все градики для нейронов

            loss_item = loss.item()
            loss_val += loss_item
            loss_my_item = lossing(pred.cpu(), label.cpu())
            loss_my += loss_my_item

            optimizer.step()  # Сделали шаг градиентным спуском с учетом градиента посчитанного loss.backward()

            acc_current = accuracy(pred.cpu(), label.cpu(), epsilon=0.1)
            acc_val += acc_current

        # смотрим какая ошибка на одной картинке loss_item
        print(f'epoch: {count}\tloss: {loss_my / 110}\tacuraccy: {acc_val / 110}')
        accuracies.append(acc_val / 110)
        losses.append(loss_my / 110)
    print(f'Full time learning : {time.time() - start_time}')
    path_name = 'model_take_profit_'+device+'.pth' if label_offset == 0 else 'model_stop_loss_'+device+'.pth'
    torch.save(model.state_dict(), path_name)




    h = np.linspace(1, 1500, 1500)

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