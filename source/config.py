import pandas as pd
import torch

def configure_display():
    torch.set_printoptions(linewidth=1000)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 500)
