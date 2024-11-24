import torch
import torch.nn.functional as F

# FGSM Attack Code
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Keep values in valid range
    return perturbed_image

# Denormalize function (to restore the original image scale)
def denorm(batch, mean=[0.1307], std=[0.3081]):
    for t, m, s in zip(batch, mean, std):
        t.mul_(s).add_(m)
    return batch
