import torch
from torchvision import models, datasets, transforms
from captum.influence import SimilarityInfluence
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='../training', transform=transform)
test_dataset = datasets.ImageFolder(root='../data', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)  # Shuffle is False for test set

activation_dir = 'activations'
os.makedirs(activation_dir, exist_ok=True)  

# NOTE: IMPLEMENT THE COSINE SIMILARITY METRIC
def cosine_similarity_metric(x, y):
    

similarity_influence = SimilarityInfluence(
    module=model,
    layers=['layer4'],
    influence_src_dataset=train_dataset,
    activation_dir=activation_dir,  # Use the variable instead of 'path_to_activation_dir'
    model_id='resnet18_example',
    similarity_metric=cosine_similarity_metric,
    batch_size=32
)

# Get a single batch from the test set
test_inputs, test_targets = next(iter(test_loader))

# Compute influence scores
influences = similarity_influence.influence(
    inputs=test_inputs,
    top_k=5,  # Adjust top_k to the number of similar instances you want
    additional_forward_args=None,
    load_src_from_disk=True  # Set to True to use precomputed activations if available
)

# Extracting the indices for the first test example's top similar training instances
layer_name = 'layer4'
top_k_indices = influences[layer_name][0][0][0].cpu().numpy()  # Indices of top similar instances for the first test example
top_k_scores = influences[layer_name][1][0][0].cpu().numpy()  # Scores of top similar instances for the first test example

# Visualize the top k indices and scores
plt.bar(range(len(top_k_indices)), top_k_scores, tick_label=top_k_indices)
plt.xlabel('Training Instance Index')
plt.ylabel('Similarity Score')
plt.title('Top Similar Training Instances for First Test Example')
plt.show()