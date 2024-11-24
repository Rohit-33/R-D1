import torch
from PIL import Image
from transformers import CLIPTokenizer
import model_loader
import pipeline
from pathlib import Path

# Device configuration
DEVICE = "cpu"  # Default is CPU. Update based on availability.

ALLOW_CUDA = False  # Set True if you want to use CUDA.
ALLOW_MPS = False   # Set True if you're on an Apple device with MPS support.

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Initialize tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # Modify path as needed.
model_file = "data/v1-5-pruned-emaonly.ckpt"  # Modify path to your model checkpoint
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Prepare input image
input_image = Image.open("images/dog.jpg").convert("RGB")  # Modify path as needed.
prompt = "A dog wearing sunglasses"
uncond_prompt = ""  # Empty string for negative prompt

# Tokenizing the prompt
C = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
uncond_C = tokenizer.encode(uncond_prompt, return_tensors='pt').to(DEVICE)

# Image Latent Mapping (ILM)
def image_latent_mapping(z0, C, models, T, Ni):
    zt = [z0]
    for t in range(T, 0, -1):
        for j in range(Ni):
            # Optimization step for null text embedding ∅t
            ∅t = torch.zeros_like(C).to(DEVICE)  # Initialize null text embedding
            ∅t -= ζ * torch.autograd.grad(
                torch.norm(z∗t - zt[-1], p=2), ∅t, retain_graph=True
            )[0]
        zt.append(zt[-1])  # Update zt with modified ∅t at each step
    return zt

# Adversarial Latent Optimization (ALO)
def adversarial_latent_optimization(zT, C, uncond_C, models, Na, μ, eta, κ=0.1):
    δ = torch.zeros_like(zT).to(DEVICE)  # Initialize perturbation
    g = torch.zeros_like(zT).to(DEVICE)  # Initialize momentum
    
    for k in range(1, Na + 1):
        # Generate the adversarial example using the perturbation
        z0_perturbed = Ω(zT + δ, C)  # Ω is the inverse process (DDIM)

        # Compute gradient and update the perturbation
        loss = compute_adversarial_loss(z0_perturbed, C, uncond_C, models)  # Compute the adversarial loss
        loss.backward()

        # Momentum update
        g = μ * g + (1 - μ) * δ.grad.sign()

        # Project perturbation to L∞ ball (within the specified range)
        δ = torch.clamp(δ + eta * g, -κ, κ)
        zT = zT + δ  # Update the latent

        print(f"Adversarial iteration {k}, Loss: {loss.item()}")
    
    return zT

# Compute the adversarial loss using a classifier Fθ
def compute_adversarial_loss(z0_perturbed, C, uncond_C, models):
    # Cross-entropy loss for classification misguidance and MSE for reconstruction
    classifier_loss = torch.nn.CrossEntropyLoss()(models.classifier(z0_perturbed), C)  # Assuming classifier model
    mse_loss = torch.nn.MSELoss()(z0_perturbed, C)  # L2 loss for reconstruction
    return classifier_loss - 0.1 * mse_loss

# Inverse process using DDIM (assuming the models have an encode/decode mechanism)
def Ω(zT, C, T, models):
    # DDIM inversion process (customize based on your model's structure)
    for t in range(T, 0, -1):
        zt = models.decode_latent(zT, t)
    return zt

# Generate the image
def generate_adversarial_image(input_image, prompt, uncond_prompt, models, T=50, Ni=10, Na=50, μ=0.9, eta=0.1):
    # Initial latent (z0) representation of the input image
    z0 = models.encode_image(input_image)  # Assuming an encode method
    
    # Perform image latent mapping
    z0_latent = image_latent_mapping(z0, C, models, T, Ni)

    # Perform adversarial optimization
    zT = adversarial_latent_optimization(z0_latent[-1], C, uncond_C, models, Na, μ, eta)
    
    # Decode the adversarial latents back to image
    adv_image = models.decode_latent(zT)
    adv_image.save("outputs/adversarial_dog.png")
    print("Adversarial image saved to outputs/")

# Run the adversarial image generation
generate_adversarial_image(input_image, prompt, uncond_prompt, models)
