import torch
from control import control
from args import args, flags

con = control()
if flags.mode == 'train':
    con.train()
elif flags.mode == 'predict':
    con.predict()