from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

from torchgeometry.losses import tversky

from medicalDataLoader import MyDataloader
import argparse
from utils import *

from UNet_Base import *
import random
import torch
import pdb

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix




class MyModel(object):
    def __init__(self, args):
        self.args = args

        # Chargement des données
        self.args.root_dir = './Data/'
        myDataLoader = MyDataloader(self.args)
        self.train_loader, self.val_loader = myDataLoader.create_labelled_dataloaders()
        self.unlabelled_loader = myDataLoader.create_unlabelled_dataloaders()
        self.num_classes = 4
        self.losses_directory = 'Results/Statistics/' + self.args.model + '/' + self.args.name
        self.model_directory = 'models/' + self.args.model + '/' + self.args.name

        # Model and optimizer
        print(" Model Name: {}".format(self.args.model))
        match self.args.model:
            case 'Unet':
                self.model = UNet(self.num_classes)
            case 'Unet':
                self.model = ...#nnUNet(self.num_classes)

        match self.args.optimizer:
            case 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                 lr=self.args.lr, 
                                                 weight_decay=5e-4, 
                                                 momentum=self.args.momentum)
            case 'Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.softMax = nn.Softmax()
        match self.args.loss:
            case 'CE':
                self.loss = nn.CrossEntropyLoss(self.model.parameters())
            case 'focal':
                self.loss = ...#nn.FocalLoss()

        if self.args.cuda.is_available():
            self.model.cuda()
            self.softMax.cuda()
            self.loss.cuda()

        if self.args.load_weights:
            if self.args.cuda:
                self.model.module.load_state_dict(self.checkpoints['model_state_dict'])
            else:
                self.model.load_state_dict(self.checkpoints['model_state_dict'])
            self.optimiser.load_state_dict(self.checkpoints['optimiser_state_dict'])

        # Statistics
        self.loss_training = []


    def training(self):
        self.model.train()
        lossEpoch = []
        mean_acc = np.array([0,0,0,0]).astype(float)
        # DSCEpoch = []
        # DSCEpoch_w = []
        num_batches = len(self.train_loader)
        
        n = 0
        ## FOR EACH BATCH
        for j, data in enumerate(self.train_loader):
            ### Set to zero all the gradients
            self.model.zero_grad()
            self.optimizer.zero_grad()

            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, _ = data
            labels = getTargetSegmentation(to_var(labels))
            images = to_var(images)

            #-- The CNN makes its predictions (forward pass)
            pred = self.model.forward(images)

            # COMPUTE THE LOSS
            CE_loss_value = self.loss(pred, labels)
            lossTotal = CE_loss_value
            # masks = torch.argmax(self.softMax(pred), dim=1)

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            lossTotal.backward()
            self.optimizer.step()
            confmat = ConfusionMatrix(task="multiclass", num_classes=4)
            confmat = confmat(pred, labels).numpy()
            accuracy = np.array([confmat[0,0]/confmat[:,0].sum(),
                        confmat[1,1]/confmat[:,1].sum(),
                        confmat[2,2]/confmat[:,2].sum(),
                        confmat[3,3]/confmat[:,3].sum(),]).astype(float)

            accuracy[accuracy==float('nan')] = 0
            mean_acc += accuracy
            n += 1
            
            # THIS IS JUST TO VISUALIZE THE TRAINING 
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}]".format(lossTotal,accuracy[0],accuracy[1],accuracy[2],accuracy[3]))
        
        mean_acc = mean_acc / n
        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)
        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}]".format(i,lossEpoch,mean_acc[0],mean_acc[1],mean_acc[2],mean_acc[3]))


    def validation(self):
        self.model.eval()
        lossValEpoch = []
        mean_val_acc = np.array([0,0,0,0]).astype(float)
        num_batches_val = len(val_loader)



    def inference(self):
        ...



def main():
    parser = argparse.ArgumentParser(description="Entrainement challenge MTI865")
    # Hyperparamètres généraux
    parser.add_argument('--name', type=str, default='modele_test'
                        help='Nom du modèle')
    parser.add_argument('--augment', action='store_true', default='False',
                        help='Augmentation des données')
    parser.add_argument('--loss', type=str, default='CE',
                        choices=['CE', 'focal'],
                        help='Choix de la loss (défaut: Unet)')
    parser.add_argument('--model', type=str, default='Unet',
                        choices=['Unet', 'nnUnet'],
                        help='Choix du modèle (défaut: Unet)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Nombre de coeurs pour les dataloaders (défaut: 0)')
    # Hyperparamètres d'entrainement
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Nombre d\'epochs pour l\'entrainement (défaut: 10)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='Epoch de départ (défaut:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='Taille des batchs pour \
                                l\'entrainement (défaut: 16)')
    parser.add_argument('--val-batch-size', type=int, default=4,
                        metavar='N', help='Taille des batchs pour la \
                                validation (défaut: 4)')
    # Hyperparamètres de l'optimiseur
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam','SGD'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate ou taux d\'apprentissage (défaut: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (défaut: 0.9)')
    # Cuda
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Authorise l\'utilisation de cuda')
    # Chargement de poids
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Chemin des poids à charger')
    # Apprentissage semi-supervisé
    parser.add_argument('--non-label', action='store_true', default=False,
                        help='Entrainement de l\'encodeur sur le dataset \
                            non labellisé')
    # Options d'évaluation
    parser.add_argument('--inference', action='store_true', default=False,
                        help='Réalise des prédictions et affiche les résultats')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda disponible :",torch.cuda.is_available())

    modele = MyModel(args)

    if args.inference is True:
        print('Inférence:')
        modele.inference()
    else :
        print('Epoch de départ:', modele.args.start_epoch)
        print('Epochs totales:', modele.args.epochs)
        for epoch in range(modele.args.start_epoch, modele.args.epochs):
            if args.non_label:
                ...
            else:
                modele.training(epoch)
                modele.validation(epoch)


if __name__ == "__main__":
   main()
