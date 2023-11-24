from nnunet.network_architecture.generic_UNet import Generic_UNet


class nnUnet(Generic_UNet):
    def __init__(self, num_classes):
        super().__init__(input_channels=1, base_num_features=64, num_classes=num_classes,num_pool=2)

    def parameters(self):
        return super().parameters()
    
    def forward(self, x):
        x = super().forward(x)[0]
        return x

#def nnUnet(num_classes):
#    from nnunet.network_architecture.generic_UNet import Generic_UNet
#    model = Generic_UNet(input_channels=1, base_num_features=64, num_classes=num_classes,num_pool=1)
#
#    for name, parameter in model.named_parameters():
#        if 'seg_outputs' in name:
#            print(f"parameter '{name}' will not be freezed")
#            parameter.requires_grad = True
#        else:
#            parameter.requires_grad = False
#
#
#    return model