import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

# Assuming your encoder and diffusion models are imported
from encoder_model import Encoder  # your Encoder model
from diffusion_model import DiffusionModel  # your diffusion model

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the necessary transformations for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load a sample image from a dataset (e.g., MNIST, CIFAR)
dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
image, label = dataset[0]

# Visualize the original image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Original Image (Label: {label})")
plt.show()

# Move image to correct device
image = image.unsqueeze(0).to(DEVICE)  # Add batch dimension

# Instantiate the encoder and diffusion models
encoder = Encoder().to(DEVICE)
encoder.eval()

diffusion_model = DiffusionModel().to(DEVICE)  # Assuming DiffusionModel is implemented
diffusion_model.eval()

# Tokenizer for CLIP model (assuming it's part of your pipeline)
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate image using the adversarial latent
def generate_image_with_fgsm(prompt, input_image, encoder, tokenizer, models, device, fgsm_epsilon=0.1):
    # Pass input_image to the pipeline for encoding, FGSM, and diffusion
    generated_image = pipeline.generate(
        prompt=prompt,
        input_image=input_image,
        encoder=encoder,
        tokenizer=tokenizer,
        models=models,
        device=device,
        fgsm_epsilon=fgsm_epsilon
    )
    
    # Convert tensor to image for visualization
    generated_image = generated_image.squeeze().cpu().numpy().astype(np.uint8)
    return Image.fromarray(generated_image)

# Define models dict (contains encoder, diffusion model, etc.)
models = {
    "clip": clip_model,  # Assuming you have a CLIP model loaded
    "diffusion": diffusion_model,
    "encoder": encoder,
}

# Example prompt
prompt = "A drawing of a cat"

# Generate the adversarial image
output_image = generate_image_with_fgsm(prompt, image, encoder, tokenizer, models, DEVICE)

# Display the generated image
output_image.show()
