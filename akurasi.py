import os
import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.stats import entropy
from PIL import Image

# Function to load Inception v3 model
def load_inception_v3():
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model

# Function to preprocess images
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# Function to extract features from images using Inception v3
def extract_features(images, model):
    features = []
    for img in images:
        img_tensor = preprocess_image(img)
        with torch.no_grad():
            feature = model(img_tensor).squeeze().cpu().numpy()
        features.append(feature)
    return np.array(features)

# Function to calculate Frechet Inception Distance (FID)
def calculate_fid(real_images, generated_images, model):
    real_features = extract_features(real_images, model)
    generated_features = extract_features(generated_images, model)
    
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FID score
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_score

# Function to calculate Frechet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

# Function to calculate Inception Score (IS)
def calculate_inception_score(images, model, batch_size=50, splits=10):
    features = extract_features(images, model)
    scores = []
    n_batches = len(images) // batch_size
    for i in range(n_batches):
        batch = features[i * batch_size:(i + 1) * batch_size]
        with torch.no_grad():
            preds = torch.softmax(torch.tensor(batch), dim=1)
            scores.extend(preds.numpy())
    
    scores = np.array(scores)
    p_y = np.mean(scores, axis=0)
    kl_divs = scores * (np.log(scores) - np.log(p_y))
    kl_divergence = np.mean(np.sum(kl_divs, axis=1))
    is_score = np.exp(kl_divergence)
    
    return is_score

# Example usage
if __name__ == "__main__":
    # Load Inception v3 model
    inception_model = load_inception_v3()

    # Example of real and generated images (replace with your data)
    real_images = [...]  # List of real images
    generated_images = [...]  # List of generated images
    
    # Calculate FID
    fid_score = calculate_fid(real_images, generated_images, inception_model)
    print(f"FID Score: {fid_score}")
    
    # Calculate IS
    is_score = calculate_inception_score(generated_images, inception_model)
    print(f"Inception Score: {is_score}")
