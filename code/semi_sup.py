import torch
import torchvision
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from Unet_pp import *
from torchvision import transforms
import medicalDataLoader
from torch.utils.data import DataLoader
from utils import *

def main():
    root_dir = './Data/'
    model = Unet_pp(4)
    checkpoints = torch.load('./models/UnetPP/testProf/best_model', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.20))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    semi_set = medicalDataLoader.MedicalImageDataset('loadsemi',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    semi_loader = DataLoader(semi_set,
                            batch_size=1,
                            num_workers=5,
                            shuffle=False)
    softMax = torch.nn.Softmax().cuda()
    save_path = './Data/train/GTsemi/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for j, data in enumerate(semi_loader):
        images, labels, paths = data
        save_path = paths[0].replace('Img-Unlabeled','GTsemi')

        images = images.cuda()
        images = to_var(images)
        
        pred = model.forward(images.float())
        pred_y = softMax(pred)

        masks = torch.argmax(pred_y, dim=1)
        masks=masks.view((256,256))

        torchvision.utils.save_image(masks.view(labels.shape[0],1,256,256).data/3.0,save_path, padding=0)

if __name__ == '__main__':
    main()