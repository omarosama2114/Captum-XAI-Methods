import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import models, transforms
from captum.concept import TCAV, Concept

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load images from a folder and apply transformations
def load_images_from_folder(folder, transform):
    images = []
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        with Image.open(img_path) as img:
            images.append(transform(img))
    return torch.stack(images)  # Converts a list of tensors to a tensor

# Load concept images
striped_images = load_images_from_folder('striped', transform)
random_images = load_images_from_folder('random', transform)

# Assume that we use a simple sequential data iterator for simplicity
striped_concept = Concept(0, "striped", iter([striped_images]))
random_concept = Concept(1, "random", iter([random_images]))

# Choose the layers where you want to interpret the concepts
layers = ['layer4']

# Initialize TCAV
tcav = TCAV(model=model,
            layers=layers,
            model_id='resnet18_example',
            save_path='./tcav_results/')

experimental_sets = [[striped_concept, random_concept]]
tcav.compute_cavs(experimental_sets)

test_images = load_images_from_folder('test_images', transform)

target_class_idx = 340  # The index of the target class 'Zebra' in ImageNet

# Create a tensor with the target class index repeated for as many times as there are test images
target_class_index = torch.tensor([target_class_idx for _ in range(test_images.size(0))])

target_class_index = torch.tensor([target_class_idx for _ in range(test_images.size(0))])
results = tcav.interpret(inputs=test_images,
                         experimental_sets=experimental_sets,
                         target=target_class_index)

def visualize_tcav_scores(results):
    # Assuming 'results' is structured as shown, and there's only one experimental set ('0-1') in this case
    exp_set_key = '0-1'  # This is your experimental set key as shown in the results
    layer_key = 'layer4'  # This is your layer key as shown in the results
    
    # Extract the 'sign_count' and 'magnitude' for the experimental set and layer
    sign_count = results[exp_set_key][layer_key]['sign_count']
    magnitude = results[exp_set_key][layer_key]['magnitude']
    
    # Labels for plotting
    concepts = ['Positive Influence', 'Negative Influence']
    
    # Plotting
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(concepts, sign_count.numpy(), color='skyblue')
    plt.title('Sign Count Score')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.bar(concepts, magnitude.numpy(), color='lightgreen')
    plt.title('Magnitude Score')
    plt.ylabel('Score')
    
    plt.suptitle('TCAV Results for Concept "Striped"')
    plt.tight_layout()
    plt.show()

visualize_tcav_scores(results)