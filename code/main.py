import argparse
import torch
from progressBar import printProgressBar
from medicalDataLoader import MyDataloader
from utils import *
from UNet_Base import *
from UNet_Boosted import *
from UNet_Higher import *
import segmentation_models_pytorch as smp
from data_augmentation import augment_data
import losses

# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchgeometry.losses import tversky
# import random
# import pdb
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score 
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

WANDB_TRACKING = False

class MyModel(object):
    def __init__(self, args):
        self.args = args

        # Chargement des données
        self.args.root_dir = './Data/'
        self.args.augment_dir = './Augment/'

        if self.args.augment is True:
            augment_data(self.args.root_dir, self.args.augment_dir)
            self.args.root_dir = self.args.augment_dir
        if self.args.on_augmented is True:
            self.args.root_dir = self.args.augment_dir

        myDataLoader = MyDataloader(self.args)
        self.train_loader, self.val_loader, self.test_loader = myDataLoader.create_labelled_dataloaders()
        self.unlabelled_loader = myDataLoader.create_unlabelled_dataloaders()
        self.num_classes = 4
        self.losses_directory = 'Results/Statistics/' + self.args.model + '/' + self.args.name
        self.model_directory = 'models/' + self.args.model + '/' + self.args.name
        
        aux_params=dict( #Params for pretrained models
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=4              # define number of output labels
        )

        # Model
        print("Nom du modèle : {}".format(self.args.model))
        match self.args.model:
            case 'Unet':
                self.model = UNet(self.num_classes)
            case 'UnetBoosted':
                self.model = UNetBoosted(self.num_classes)
            case 'UnetHigher':
                self.model = UNetHigher(self.num_classes)
            case 'pretrained':
                self.model = smp.UnetPlusPlus('resnet50', classes=4, aux_params=aux_params,in_channels=1)   
        print("Nombre de paramètres: {0:,}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Loss
        self.softMax = nn.Softmax()
        self.dice = losses.dice_loss()
        # self.dice = nn.KLDivLoss()
        # self.dice = losses.MyCenterLoss()
        self.loss2_factor = 0.1
        match self.args.loss:
            case 'CE':
                if self.args.loss_weights is None :
                    self.loss = nn.CrossEntropyLoss()
                else:
                    w = torch.FloatTensor(self.args.loss_weights)
                    self.loss = nn.CrossEntropyLoss(weight=w)
            case 'Dice':
                if self.args.loss_weights is not None :
                    w = self.args.loss_weights
                    self.loss = losses.dice_loss(weight=w)
                else:
                    self.loss = losses.dice_loss()
            case 'KLDiv':
                self.loss = nn.KLDivLoss()

        # Optimizer
        match self.args.optimizer:
            case 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                 lr=self.args.lr, 
                                                 weight_decay=5e-4, 
                                                 momentum=self.args.momentum)
            case 'Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=self.args.lr,
                                                  betas=[.9,.999],
                                                  weight_decay=0)
            case 'NAdam':
                self.optimizer = torch.optim.NAdam(self.model.parameters(),
                                                   lr=self.args.lr,
                                                   betas=[.9,.999],
                                                   weight_decay=0,
                                                   momentum_decay=0.004)
            case 'RAdam':
                self.optimizer = torch.optim.RAdam(self.model.parameters(),
                                                   lr=self.args.lr,
                                                   betas=[.9,.999],
                                                   weight_decay=0)
            case 'Adamax':
                self.optimizer = torch.optim.Adamax(self.model.parameters(),
                                                    lr=self.args.lr,
                                                    betas=[.9,.999],
                                                    weight_decay=0)

        if self.args.cuda is True:
            self.model.cuda()
            self.softMax.cuda()
            self.loss.cuda()

        if not self.args.load_weights is None:
            self.checkpoints = torch.load('./models/'+self.args.model+'/'+self.args.load_weights+'/best_model')
            if self.args.cuda is True:
                self.model.module.load_state_dict(self.checkpoints['model_state_dict'])
            else:
                self.model.load_state_dict(self.checkpoints['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoints['optimizer_state_dict'])
            
            if 'best_loss' in self.checkpoints:
                self.best_loss = self.checkpoints['best_loss']
            else:
                self.best_loss = 1000

        else:
            self.best_loss = 1000

        # Statistics
        self.loss_training = []
        self.loss_validation = []


    def training(self, epoch):
        self.model.train()
        loss_epoch = []
        mean_acc = np.zeros(4).astype(float)
        mean_dice = np.zeros(4).astype(float)
        num_batches = len(self.train_loader)
        
        ## FOR EACH BATCH
        for j, data in enumerate(self.train_loader):
            ### Set to zero all the gradients
            self.model.zero_grad()
            self.optimizer.zero_grad()

            ## GET IMAGES, LABELS and IMG NAMES
            images, targets, _ = data
            labels = to_var(getTargetSegmentation(targets))
            images = to_var(images)

            #-- The CNN makes its predictions (forward pass)
            pred = self.model.forward(images)
            # pred = self.softMax(self.model.forward(images))

            # COMPUTE THE LOSS
            loss_value = self.loss(pred, labels)
            dice = self.loss2_factor*self.dice(pred, labels)
            # loss_value = loss_value + dice

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            loss_value.backward()
            self.optimizer.step()

            accuracy = evaluation(torch.argmax(pred, dim=1), labels, self.num_classes)
            mean_acc += accuracy
            mean_dice += dice.cpu().data.numpy()

            #Eval_HD = np.zeros(4)
            #for i in range(0, 4):
            #    Eval_HD[i] = losses.HausdorffLoss(pred[0,i,:,:], labels[0,i,:,:])
            if WANDB_TRACKING:
                try:
                    wandb.log({"loss": loss_value,"accuracy0": accuracy[0],"accuracy1": accuracy[1],"accuracy2": accuracy[2],"accuracy3": accuracy[3],"dice":dice,"epoch":epoch})
                except:
                    pass

            # THIS IS JUST TO VISUALIZE THE TRAINING 
            loss_epoch.append(loss_value.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(epoch),
                             length=15,
                             suffix=" Loss: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}], Dice: {:.4f}"    
                             .format(loss_value.cpu(),accuracy[0],accuracy[1],accuracy[2],accuracy[3],dice))

        mean_dice = mean_dice / num_batches
        mean_acc = mean_acc / num_batches
        mean_acc[np.isnan(mean_acc)] = 0
        loss_epoch = np.asarray(loss_epoch).mean()

        self.loss_training.append(loss_epoch)
        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}, Mean Acc: [{:.4f},{:.4f},{:.4f},{:.4f}], Dice: {:.4f}"
                             .format(epoch,loss_epoch,mean_acc[0],mean_acc[1],mean_acc[2],mean_acc[3],mean_dice.mean()))


    def validation(self, epoch):
        self.model.eval()
        loss_epoch = []
        mean_acc = np.zeros(4).astype(float)
        mean_dice = np.zeros(4).astype(float)
        num_batches = len(self.val_loader)

        for j, data in enumerate(self.val_loader):
            images, targets, _ = data
            labels = getTargetSegmentation(to_var(targets))
            images = to_var(images)
            pred = self.model.forward(images.float())
            # pred = self.softMax(self.model.forward(images.float()))
            loss_value = self.loss(pred, labels)
            dice = self.loss2_factor*self.dice(pred, targets)
            # loss_value = loss_value + dice
            accuracy = evaluation(pred, labels, self.num_classes)
            mean_acc += accuracy
            mean_dice += dice.cpu().data.numpy()

            #Eval_HD = np.zeros(4)
            #for i in range(0, 4):
            #    Eval_HD[i] = losses.HausdorffLoss(pred[0,i,:,:], labels[0,i,:,:])
            if WANDB_TRACKING:
                try:
                    wandb.log({"val_loss": loss_value,"val_accuracy0": accuracy[0],"val_accuracy1": accuracy[1],"val_accuracy2": accuracy[2],"val_accuracy3": accuracy[3],"val_dice": dice,"epoch":epoch})
                except:
                    pass

            loss_epoch.append(loss_value.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Validation] Epoch: {} ".format(epoch),
                             length=15,
                             suffix=" Loss: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}], Dice: {:.4f}"
                             .format(loss_value,accuracy[0],accuracy[1],accuracy[2],accuracy[3],dice))
        
        mean_dice = mean_dice / num_batches
        mean_acc = mean_acc / num_batches
        mean_acc[np.isnan(mean_acc)] = 0

        loss_epoch = np.asarray(loss_epoch).mean()
        self.loss_validation.append(loss_epoch)

        is_saved = False
        if loss_epoch < self.best_loss:
            is_saved = True
            self.save(epoch)
        
        printProgressBar(num_batches, num_batches,
                             done="[Validation] Epoch: {}, LossG: {:.4f}, Mean Acc: [{:.4f},{:.4f},{:.4f},{:.4f}], Dice: {:.4f}, Save: {}"
                             .format(epoch,loss_epoch,mean_acc[0],mean_acc[1],mean_acc[2],mean_acc[3],mean_dice.mean(),is_saved))


    def inference(self):
        self.model.eval()
        loss_epoch = []
        mean_acc = np.array([0,0,0,0]).astype(float)
        num_batches = len(self.val_loader)

        for j, data in enumerate(self.val_loader):
            images, labels, _ = data
            labels = getTargetSegmentation(to_var(labels))
            images = to_var(images)
            pred = self.softMax(self.model.forward(images.float()))
            loss_value = self.loss(pred, labels)
            accuracy = evaluation(pred, labels, self.num_classes)
            mean_acc += accuracy

            loss_epoch.append(loss_value.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Inference]",
                             length=15,
                             suffix=" Loss: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}] ".format(loss_value,accuracy[0],accuracy[1],accuracy[2],accuracy[3]))
            
            for i, image in enumerate(images.numpy()):
                fig = plt.figure()
                plt.subplot(1,3,1).set_title('Image')
                plt.imshow(image[0])
                plt.colorbar()
                plt.subplot(1,3,2).set_title('Label')
                plt.imshow(labels.numpy()[i])
                plt.colorbar()
                plt.subplot(1,3,3).set_title('Prédiction')
                plt.imshow(torch.argmax(pred, dim=1).numpy()[i])
                plt.colorbar()
                fig.suptitle('Loss: {:.4f}, Acc: [{:.4f},{:.4f},{:.4f},{:.4f}]'.format(
                    loss_value,accuracy[0],accuracy[1],accuracy[2],accuracy[3]))
                plt.show()
        
        mean_acc = mean_acc / num_batches
        mean_acc[np.isnan(mean_acc)] = 0
        loss_epoch = np.asarray(loss_epoch).mean()
        
        printProgressBar(num_batches, num_batches,
                             done="[Inference], LossG: {:.4f}, Mean Acc: [{:.4f},{:.4f},{:.4f},{:.4f}]".format(loss_epoch,mean_acc[0],mean_acc[1],mean_acc[2],mean_acc[3]))




    def save(self, epoch):
        self.best_loss = self.loss_validation[epoch]
        self.best_epoch = epoch
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        torch.save({'epoch': epoch,
                    'batch_size':self.args.batch_size,
                    'batch_size_val':self.args.val_batch_size,
                    'lr':self.args.lr,
                    'best_loss':self.best_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.model_directory + '/best_model')
        
        if not os.path.exists(self.losses_directory):
            os.makedirs(self.losses_directory)
        np.save(os.path.join(self.losses_directory, 'losses.npy'),
                np.array((self.loss_training, self.loss_validation)))
        

    def display_losses(self):
        try:
            losses = np.load(os.path.join(self.losses_directory, 'losses.npy'))
            plt.plot(losses[0])
            plt.plot(losses[1])
            plt.show()
        except:
            print('Pas de losses pour le modèle {}'.format(self.args.name))



def main():
    parser = argparse.ArgumentParser(description="Entrainement challenge MTI865")
    # Hyperparamètres généraux
    parser.add_argument('--name', type=str, default='modele_test',
                        help='Nom du modèle')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Augmentation des données et entraînement sur ces données augmentées')
    parser.add_argument('--on-augmented', action='store_true', default=False,
                        help='Entraînement sur les données augmentées au préalable')
    parser.add_argument('--loss', type=str, default='CE',
                        choices=['CE', 'Dice'], help='Choix de la loss (défaut: Unet)')
    parser.add_argument('--loss-weights', nargs='+', type=float, default=None, action='store',
                        help='Liste de 4 poids (défaut: None)')
    parser.add_argument('--model', type=str, default='Unet',
                        choices=['Unet', 'UnetBoosted', 'UnetHigher', 'pretrained'],
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
    parser.add_argument('--val-batch-size', type=int, default=8,
                        metavar='N', help='Taille des batchs pour la \
                                validation (défaut: 8)')
    # Hyperparamètres de l'optimiseur
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam','SGD','NAdam','RAdam','Adamax'])
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
    # Options d'évaluation
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Activer ou non le trackeur wandb')

    args = parser.parse_args()
    save_args_to_sh(args)

    args.cuda = args.cuda and torch.cuda.is_available()
    print("Cuda disponible :",torch.cuda.is_available())

    global WANDB_TRACKING
    WANDB_TRACKING = args.wandb

    model = MyModel(args)

    # Setup Wandb
    if WANDB_TRACKING:
        run = wandb.init(
            # Set the project where this run will be logged
            project="projet_segmentation",
            name="UnetBase",
            resume="allow", # See https://docs.wandb.ai/guides/runs/resuming
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
        )

    if args.inference is True:
        print('Inférence:')
        model.inference()
    else :
        print('Epoch de départ:', model.args.start_epoch)
        print('Epochs totales:', model.args.epochs)
        for epoch in range(model.args.start_epoch, model.args.epochs):
            model.training(epoch)
            model.validation(epoch)
    if WANDB_TRACKING:
        try:
            wandb.finish()
        except:
            pass
    model.display_losses()


if __name__ == "__main__":
   main()
