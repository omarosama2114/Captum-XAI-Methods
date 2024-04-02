import numpy as np
import matplotlib.pyplot as plt
import torch
from labels import imagenet_labels
from PIL import Image
from torchvision import models, transforms
from captum.attr import GuidedGradCam
import cv2

# Load the model
model = models.resnet18(pretrained=True)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img = Image.open("cat.jpg")

# Apply transformations
input_img = transform(img).unsqueeze(0)  # Add batch dimension

# Define the target layer for Guided Grad-CAM
target_layer = model.layer4[-1].conv2

# Perform a forward pass to get the logits
outputs = model(input_img)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(outputs, dim=1)

# Get the top class and the corresponding index
max_prob, class_idx = torch.max(probabilities, 1)

predicted_label = imagenet_labels[class_idx.item()]
print(f"The predicted class index {class_idx.item()} corresponds to the label '{predicted_label}'.")

guided_gc = GuidedGradCam(model, target_layer)
# Compute the Guided Grad-CAM attributions
attributions_ggc = guided_gc.attribute(input_img, target=class_idx)

# Normalize the attribution map
attr = attributions_ggc.squeeze().cpu().detach().numpy()
attr = np.amax(attr, axis=0)
attr = attr - np.min(attr)   # Shift the attribution so that the lowest value is 0.
attr = attr / np.max(attr)   # Scale the attribution so that the highest value is 1.

# Apply the colormap to the normalized attribution map
# The output of plt.cm.hot is (M, N, 4). We take only the first three channels for RGB.
heatmap = plt.cm.hot(attr)[:, :, :3]
heatmap = np.uint8(255 * heatmap)  # Convert to uint8

# Prepare the original image by reversing the normalization
input_img_np = input_img.squeeze().cpu().detach().numpy()
input_img_np = np.transpose(input_img_np, (1, 2, 0))
input_img_np = input_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
input_img_np = np.clip(input_img_np, 0, 1)

# Overlay the heatmap on top of the original image
alpha = 0.1  # transparency for the heatmap
overlay_img = heatmap * alpha + input_img_np * (1 - alpha)
overlay_img = np.clip(overlay_img, 0, 1)

# Display the image
plt.imshow(overlay_img)
plt.axis('off')  # Hide the axes
plt.title(f"The predicted class index {class_idx.item()} corresponds to the label '{predicted_label}'.")
plt.show()
