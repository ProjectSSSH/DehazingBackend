from PIL import Image
import numpy as np
import torch
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dark_channel_prior(image, window_size=15):
    dark_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    num_pixels = image.shape[0] * image.shape[1]
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argsort(dark_channel.ravel())[-num_brightest:]
    brightest_pixels = image.reshape(-1, 3)[indices]
    A = brightest_pixels.mean(axis=0)
    return A

def transmission_estimate(image, A, omega=0.95, window_size=15):
    norm_image = image / A
    transmission = 1 - omega * dark_channel_prior(norm_image, window_size)
    return transmission

def recover_image(image, transmission, A, t_min=0.1):
    transmission = np.maximum(transmission, t_min)
    transmission = transmission[..., np.newaxis]
    J = (image - A) / transmission + A
    J = np.clip(J, 0, 1)
    return J

# def load_image(path, size=(256, 256)):
#     try:
#         image = Image.open(path).convert('RGB')
#         image = image.resize(size)
#         image = np.asarray(image) / 255.0
#     except Exception as e:
#         raise ValueError(f"Error loading image: {e}")
#     return image
#     # image = Image.open(path).convert('RGB')
#     # image = image.resize(size)
#     # image = np.asarray(image) / 255.0
#     # return image

def load_image(path):
    try:
        image = Image.open(path).convert('RGB')
        image = np.asarray(image) / 255.0
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    return image

def image_to_tensor(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
    elif len(image.shape) == 2:  # Grayscale image
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unsupported image dimensions")
    return image

def tensor_to_image(tensor):
    image = tensor.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    image = np.clip(image, 0, 1)
    return image


def run_inference(model, image_path):
    # Load the hazy image
    hazy_image = load_image(image_path)  # Resize if necessary
    hazy_tensor = image_to_tensor(hazy_image).to(device)

    # Compute the dark channel prior and other features
    dark_channel = dark_channel_prior(hazy_image)
    A = atmospheric_light(hazy_image, dark_channel)
    transmission = transmission_estimate(hazy_image, A)

    # Create a tensor for the dark channel prior
    dcp_tensor = image_to_tensor(transmission).to(device)

    # Run the model to get the dehazed output
    with torch.no_grad():
        dehazed_tensor = model(hazy_tensor, dcp_tensor)

    return hazy_tensor, dehazed_tensor