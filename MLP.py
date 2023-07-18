import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
from load_data import dataload_csv, dataload_xlsx, PreDataLoader
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
        return out

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    # print(answer.shape)
    # print(answer)
    # print(answer.sum())
    # return answer.sum()
    return answer.mean()

if __name__ == '__main__':

    """
    Date,Open,High,Low,Close,EMA200,Assets,Take_profit,Stop_loss
    """
    # TODO: выложить на гит и расфорчить для одного выхода.
    # TODO: Отдебажить с мнистом (сравнить).
    # TODO: 1) - Дебагинг НС в python.
    batch_size = 25 # Кол-во элементов в баче
    candle_count = 50 # Кол-во отсматриваемых свечей в баче
    hidden_layer_size = 75 # Размер скрытого слоя
    output_layer_size = 2 # размер выходного слоя
    info_label_size = 4 # Кол-во столбцов с параметрами свечи
    correct_label_size = 2 # Кол-во столбцов с ответами в датасете
    start_position = 200 # Начальная позиция датасета
    stop_position = 360 # Конечная позиция датасета

    candles_params_count = 3 # Кол-во столбцов с параметрами свечи
    additional_params_count = 1 # Дополнительный столбец с параметрами свечи

    dataset_MSFT = dataload_xlsx('test')  # Грузим датасет из файла 'test'

    DL = PreDataLoader(data=dataset_MSFT, pred_size=info_label_size, label_size=correct_label_size, candle_count=candle_count, start=start_position, stop=stop_position)
    DL.create_exit_data()
    batches = DL.get_data()
    #print(batches)
    dataset = NSDAQDataSet(batches)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=1
    )

    model = NN_Nasdaq(input_size=candle_count * candles_params_count + additional_params_count, hidden_size=hidden_layer_size, output_size=output_layer_size)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    device = 'cpu'  # 'cpu'
    model = model.to(device)
    loss_func = loss_func.to(device)

    use_amp = True

    epochs = 15
    counter = 0
    start_time = time.time()
    # TODO: Отрефакторить
    for epoch in range(epochs):
        loss_val = 0
        # acc_val = 0
        epoch_time = time.time()
        counter = 0

        for info, label in (pbar := tqdm(dataloader)):
            #print(f"{counter*25}-{(counter+1)*25} data")
            counter+=1
            optimizer.zero_grad() # Обнуляем градиенты, чтобы они не помешали нам на прогоне новой картинки
            info = info.to(device)  # Перенесли свечи на GPU
            # label = F.one_hot(label,10).float()
            label = label.to(device)  # Перенесли таргет на GPU

            with autocast(use_amp, dtype=torch.float16):

                pred = model(info)

                loss = loss_func(pred, label) # посчитали ошибку (значение в label - значение полученое нашим model(img))

            loss.backward() # Прошелись по всему графу вычислений и посчитали все градики для нейронов
            # обратное распространение не оборачиваем в автокаст, чтобы его точность не понижалась до флот16 с флот32.
            loss_item = loss.item()
            loss_val += loss_item

            # Получение текущих значений весов
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter name: {name}")
            #         print(f"Weights: {param.data}")
            #         print()

            optimizer.step() # Сделали шаг градиентным спуском с учетом градиента посчитанного loss.backward()

            #acc_current = accuracy(pred.cpu(), label.cpu())  #pred.cpu().float(), label.cpu().float()
            #acc_val += acc_current

            pbar.set_description(f'epoch: {epoch+1}\tloss: {loss_item:.5f}') # смотрим какая ошибка на одном баче  \taccuracy: {acc_current:.3f}
            #if counter==3:
            #   break
        # print(loss_val/100) # средняя ошибка после одной эпохи обучения
        # print(acc_val/len(dataloader))
        #print(f'Time for {epoch+1} epoch: {(time.time() - epoch_time):.1f}')

    print(f'Full time learning: {(time.time() - start_time):.1f}')
    torch.save(model.state_dict(), 'model.pth')

