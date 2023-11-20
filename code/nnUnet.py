from nnunet.network_architecture.generic_UNet import Generic_UNet

def nnUnet(num_classes):
    from nnunet.network_architecture.generic_UNet import Generic_UNet
    model = Generic_UNet(input_channels=1, base_num_features=64, num_classes=num_classes,num_pool=1)

    for name, parameter in model.named_parameters():
        if 'seg_outputs' in name:
            print(f"parameter '{name}' will not be freezed")
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


    return model