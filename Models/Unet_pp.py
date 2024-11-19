import segmentation_models_pytorch as smp
from torch import nn

aux_params=dict( #Paramètres pour le modèle
            pooling='max',
            dropout=0.5,
            activation= nn.ReLU,
            classes=4
        )

# Création du modèle, on va chercher l'architecture grâce à segmentation_models_pytorch
class Unet_pp(smp.UnetPlusPlus):
    def __init__(self, num_classes):
        super().__init__('resnet34',encoder_weights='imagenet', classes=num_classes, aux_params=aux_params,in_channels=1)

    def parameters(self):
        return super().parameters()
    
    def forward(self, x):
        x = super().forward(x)[0]
        return x