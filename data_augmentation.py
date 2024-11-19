import os
import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise
from random import randrange

def augment_data(input_path, output_path, img_folder='Img', gt_folder='GT',semisup=False, rotation_range=40, translation_range=(20, 20), noise_level=0.01,num_aug=1):
    """
    Augmente les données dans input_path et les sauvegarde dans output_path.

    Parameters:
        input_path (str): Path vers le dossier contenant les images d'origine.
        output_path (str): Path vers le dossier où sauvegarder les nouvelles images.
        img_folder (str): Nom du dossier contenant les images.
        gt_folder (str): Nom du dossier contenant les masques de segmentation (GT), si il n'existe pas on met n'importe quoi tant que ce n'est pas un nom de dossier existant.
        rotation_range (int): Plage de rotation en degrés.
        translation_range (tuple): Plage de translation (tx, ty).
        noise_level (float): Niveau de bruit à ajouter aux images.
    """

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)
    print(f"********Début de l'augmentation")
    # Dossiers de données
    data_folders = ['train', 'val', 'test']

    for folder in data_folders:
        input_folder_path = os.path.join(input_path, folder)
        output_folder_path = os.path.join(output_path, folder)

        # Liste des fichiers dans le dossier img_folder
        files = os.listdir(os.path.join(input_folder_path, img_folder))

        for file in files:
            # Charger l'image
            image_path = os.path.join(input_folder_path, img_folder, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # print(f"********Chargement de l'image {image_path}")

            if image is None:
            #     print(f"********Erreur de chargement de l'image : {image_path}")
                break

            # Charger le masque (label) si présent
            mask_path = os.path.join(input_folder_path, gt_folder, file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
            for i in range(num_aug):
                # Augmentation des données
                augmented_image, augmented_mask = apply_augmentation(image, mask, rotation_range, translation_range, noise_level)

                # Sauvegarder les images augmentées dans le dossier de sortie
                os.makedirs(os.path.join(output_folder_path, img_folder), exist_ok=True)
                os.makedirs(os.path.join(output_folder_path, gt_folder), exist_ok=True)
                cv2.imwrite(os.path.join(output_folder_path, img_folder, f'{file[:-4]}_augmented'+str(i)+'.png'), augmented_image)
                cv2.imwrite(os.path.join(output_folder_path, img_folder, f'{file[:-4]}.png'), image)
                if augmented_mask is not None:
                    cv2.imwrite(os.path.join(output_folder_path, gt_folder, f'{file[:-4]}_augmented'+str(i)+'.png'), augmented_mask)
                    cv2.imwrite(os.path.join(output_folder_path, gt_folder, f'{file[:-4]}.png'), mask)

    if semisup: # Si on utilise les pseudo-labels, on veut refaire la même chose pour les données non labellisées
        input_folder_path = os.path.join(input_path, 'train')
        output_folder_path = os.path.join(output_path, 'train')
        img_folder_unlabeled = 'Img-Unlabeled'
        gt_folder_unlabeled = 'GTsemi'
        files = os.listdir(os.path.join(input_folder_path, img_folder_unlabeled))
        for file in files:
           # Charger l'image
           image_path = os.path.join(input_folder_path, img_folder_unlabeled, file)
           image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

           if image is None:
               break

           mask_path = os.path.join(input_folder_path, gt_folder_unlabeled, file)
           mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
           for i in range(num_aug):
               # Augmentation des données
               augmented_image, augmented_mask = apply_augmentation(image, mask, rotation_range, translation_range, noise_level)
               # Sauvegarder les images augmentées dans le dossier de sortie
               os.makedirs(os.path.join(output_folder_path, img_folder_unlabeled), exist_ok=True)
               os.makedirs(os.path.join(output_folder_path, gt_folder_unlabeled), exist_ok=True)
               cv2.imwrite(os.path.join(output_folder_path, img_folder_unlabeled, f'{file[:-4]}_augmented'+str(i)+'.png'), augmented_image)
               cv2.imwrite(os.path.join(output_folder_path, img_folder_unlabeled, f'{file[:-4]}.png'), image)
               if augmented_mask is not None:
                   cv2.imwrite(os.path.join(output_folder_path, gt_folder_unlabeled, f'{file[:-4]}_augmented'+str(i)+'.png'), augmented_mask)
                   cv2.imwrite(os.path.join(output_folder_path, gt_folder_unlabeled, f'{file[:-4]}.png'), mask)
    
def apply_augmentation(image, mask, rotation_range, translation_range, noise_level):
    # Rotation, translation, et bruit aléatoires
    apply_rotation = False
    apply_translation = False
    apply_noise = False
    idx_aug = randrange(3)
    match idx_aug:
        case 0:
            apply_rotation = True
        case 1:
            apply_translation = True
        case 2:
            apply_noise = True

    # Rotation aléatoire
    if apply_rotation:
        angle = np.random.uniform(-rotation_range, rotation_range)
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1), (image.shape[1], image.shape[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1), (mask.shape[1], mask.shape[0]))

    # Translation aléatoire
    if apply_translation:
        tx = np.random.uniform(-translation_range[0], translation_range[0])
        ty = np.random.uniform(-translation_range[1], translation_range[1])
        image = cv2.warpAffine(image, np.float32([[1, 0, tx], [0, 1, ty]]), (image.shape[1], image.shape[0]))
        if mask is not None:
            mask = cv2.warpAffine(mask, np.float32([[1, 0, tx], [0, 1, ty]]), (mask.shape[1], mask.shape[0]))

    # Ajout de bruit
    if apply_noise:
        # Ajout de bruit salt-and-pepper avec scikit-image
        noise = random_noise(image, mode='s&p', amount=noise_level)
        
        # Ajuster l'échelle du bruit pour être dans la plage [0, 255]
        scaled_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 255.0

        # Ajouter le bruit à l'image
        image = np.clip(image + scaled_noise, image.min(), image.max())


    #Si le masque est fourni, on le retourne avec les mêmes transformations (hormis le bruit) sinon on retourne None
    return image, mask if mask is not None else None
    


# Exemple d'utilisation
# input_path = 'chemin/vers/dossier/data'
# output_path = 'chemin/vers/dossier/data_augmented'
# augment_data(input_path, output_path)
