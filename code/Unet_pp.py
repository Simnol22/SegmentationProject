import segmentation_models_pytorch as smp
from torch import nn

aux_params=dict( #Params for pretrained models
            pooling='max',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation= nn.ReLU,      # activation function, default is None
            classes=4              # define number of output labels
        )

class Unet_pp(smp.UnetPlusPlus):
    def __init__(self, num_classes):
        super().__init__('resnet50', classes=num_classes, aux_params=aux_params,in_channels=1)

    def parameters(self):
        return super().parameters()
    
    def forward(self, x):
        x = super().forward(x)[0]
        return x