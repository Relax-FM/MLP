from enum import Enum

class Labels(Enum):
    take_profit = ['take_profit', 'tp', 't_p']
    stop_loss = ['stop_loss', 'sl', 's_l']

def get_label(param):
    name = param
    if name is None:
        raise NotImplementedError(
            'Label name is None. Please, add to config file')
    name = name.lower()

    label = None
    if name in Labels.take_profit.value :
        label = 0
        print('take_profit version')

    elif name in Labels.stop_loss.value :
        label = 1
        print('stop_loss version')
    else:
        raise NotImplementedError(
            f'Label [{name}] is not recognized. labels.py doesn\'t know {[name]}')

    return label