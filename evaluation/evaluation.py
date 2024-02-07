import torch
from torchvision import transforms as trn
from torchvision import models
import os
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm

import pandas as pd
from evaldataset import ADataset
import wavemix
from wavemix.classification import WaveMix


# evaluation on new dataset

# choose pretrained model
pretrained_model = 'finetune'  # 'finetune' or 'places365'
# mdoel architecture
arch = 'wavemix'  # 'densenet161'  'resnet50'  'wavemix'
# test dataset
csv_name = 'Artwork1.csv' 
root_add = './Artwork/p'   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if arch == 'wavemix':
    model = WaveMix(
        num_classes = 365,
        depth = 12,
        mult = 2,
        ff_channel = 256,
        final_dim = 256,
        dropout = 0.5,
        level = 2,
        initial_conv = 'pachify',
        patch_size = 8

    )

    if pretrained_model == 'places365':
        url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/Places365/places365_54.94.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=device))

    elif pretrained_model == 'finetune':
        model_file = f'{arch}_best.pth.tar'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()


elif arch == 'resnet50' or arch == 'resnet18' or arch == 'densenet161':

    model_file = ''
    if pretrained_model == 'places365':
        # Load the pre-trained weights
        model_file = f'{arch}_places365.pth.tar'
        # if not os.access(model_file, os.W_OK):
        #     weight_url = f'http://places2.csail.mit.edu/models_places365/{model_file}'
        #     os.system(f'wget {weight_url}')

    elif pretrained_model == 'finetune':
        model_file = f'{arch}_best.pth.tar'

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    if arch == 'resnet50' or arch == 'resnet18':
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    elif arch == 'densenet161':
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k,'norm.','norm'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
        model.load_state_dict(state_dict,strict=False)

    model.to(device)
    model.eval()


centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


artwork_dataset = ADataset(csv_file=csv_name, root_dir=root_add, transform=centre_crop)

labels_list = []
predictions_list = []
top1_accuracy_list = []
top5_accuracy_list = []

# Create a data loader for your custom dataset
custom_loader = torch.utils.data.DataLoader(artwork_dataset, batch_size=32, shuffle=False)

total_images = len(custom_loader.dataset)
for data in tqdm(custom_loader, total=len(custom_loader)):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.topk(outputs, k=5, dim=1)  # Get the top-5 predictions
        _, top1_pred = torch.max(outputs, 1)  # Get the top-1 prediction

    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(preds.cpu().numpy())
    top1_accuracy_list.extend((top1_pred == labels).cpu().numpy())
    top5_accuracy_list.extend([(labels[i] in preds[i]) for i in range(len(labels))])

# Create a DataFrame to store the data
data = pd.DataFrame({
    'Label': labels_list,
    'Prediction': [list(p) for p in predictions_list],
    'Top1_Accuracy': top1_accuracy_list,
    'Top5_Accuracy': top5_accuracy_list
})

# Save data
data.to_csv('Artwork_{}_{}.csv'.format(arch, pretrained_model), index=False)  # artwork_dataset_results.csv

top1_accuracy = data['Top1_Accuracy'].mean()
top5_accuracy = data['Top5_Accuracy'].mean()

print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
