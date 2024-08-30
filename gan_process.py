import torch
from model import GANGenerator, CGANGenerator, DCGANGenerator, WGANGenerator, WGANGPGenerator
import os
import uuid
from torchvision.utils import save_image

def generate_image_based_on_samples(selected_images, num_images, model_type, n_classes=10):
    latent_dim = 100
    img_shape = (3, 350, 350)
    img_shape2 = (3, 200, 200)
    channels = 3
    img_size = 100
    features_g = 16

    # Choose the appropriate generator model
    if model_type == 'GAN':
        generator = GANGenerator(latent_dim, img_shape)
        generator.load_state_dict(torch.load('GAN-G.pth', map_location=torch.device('cpu')))
    elif model_type == 'CGAN':
        generator = CGANGenerator(latent_dim, img_shape2, n_classes)
        generator.load_state_dict(torch.load('CGAN-G.pth', map_location=torch.device('cpu')))
    elif model_type == 'DCGAN':
        generator = DCGANGenerator(latent_dim, img_size, channels)
        generator.load_state_dict(torch.load('DCGAN-G.pth', map_location=torch.device('cpu')))
    elif model_type == 'WGAN':
        generator = WGANGenerator(latent_dim, img_shape2)
        generator.load_state_dict(torch.load('WGAN-G.pth', map_location=torch.device('cpu')))
    elif model_type == 'WGAN-GP':
        generator = WGANGPGenerator(latent_dim, channels, features_g)
        # Load only the generator's state dict
        checkpoint = torch.load('wgan_gp_model_final.pth', map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        raise ValueError("Unknown model type")

    generator.eval()

    # Generate images
    z = torch.randn(num_images, latent_dim)
    if model_type == 'CGAN':
        labels = torch.randint(0, n_classes, (num_images,))
        generated_images = generator(z, labels)
    elif model_type == 'WGAN-GP':
        z = z.view(num_images, latent_dim, 1, 1)  # Reshape z for ConvTranspose2d
        generated_images = generator(z)
    else:
        generated_images = generator(z)

    # Save generated images
    generated_image_paths = []
    for i in range(num_images):
        unique_filename = f"{model_type}_{uuid.uuid4().hex}.png"
        generated_image_path = os.path.join('static', 'hasil', unique_filename)
        save_image(generated_images[i], generated_image_path)
        generated_image_paths.append(generated_image_path)

    return generated_image_paths
