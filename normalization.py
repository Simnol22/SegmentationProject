import torch

def normalize_images(images):
    """
    Normalise (moyenne / écart-type) toutes les images indépendamment du batch.

    Parameters:
        images (torch.Tensor): Un tensor contenant les images de shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Les images normalisées.
    """
    #dim = (2,3) signifie qu'on utilise uniquement les 3e et 4e dimensions du tensor donc height et width
    mean = images.mean(dim=(2, 3), keepdim=True)
    std = images.std(dim=(2, 3), keepdim=True)

    normalized_images = (images - mean) / (std + 1e-7)  # Ajout d'une petite constante pour éviter une division par zéro

    return normalized_images


def normalize_images_per_batch(images):
    """
    Normalise (moyenne / écart-type) chaque batch d'images.

    Parameters:
        images (torch.Tensor): Un tensor contenant les images de shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Les images normalisées.
    """
   
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), keepdim=True)

    normalized_images = (images - mean) / (std + 1e-7)  # Ajout d'une petite constante pour éviter une division par zéro

    return normalized_images


def min_max_normalize_images(images):
    """
    Normalise les valeurs de pixel des images en utilisant la normalisation min-max.

    Parameters:
        images (torch.Tensor): Un tensor contenant les images de shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Les images normalisées.
    """

    min_images = images.min()
    max_images = images.max()

    normalized_images = (images - min_images) / (max_images - min_images + 1e-7)  
    # Ajout d'une petite constante pour éviter de diviser par zéro

    return normalized_images
