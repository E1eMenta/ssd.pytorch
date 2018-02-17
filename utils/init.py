import torch.nn.init as init
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Conv2d":
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def model_init(model):
    model.apply(weights_init)