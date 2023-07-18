import yfinance as fin
import pandas as pd
import openpyxl
import datetime as dt

Days = 200 # Кол-во дней в датасете
EMA_N = 200 # Кол-во дней в ЕМА
num_candle = Days+EMA_N

ticker = "MSFT" # Названия тикера который отслеживаем

hist = fin.download(ticker, period=f'{num_candle}d', interval='1d')
hist.reset_index(inplace=True)

del hist['Adj Close']
del hist['Volume']

hist[f'EMA{EMA_N}'] = hist['Close'].ewm(span=EMA_N, adjust=False).mean() # Считаем нужный ЕМА

hist['Assets'] = 0
hist['Take_profit'] = 0
hist['Stop_loss'] = 0

hist['Date'] = hist['Date'].dt.strftime('%d.%m.%Y')
hist = hist.set_index('Date')

#hist.to_csv('data.csv')
hist.to_excel('data.xlsx', sheet_name=f'{ticker}_{dt.date.today().strftime("%d-%m-%Y")}')
