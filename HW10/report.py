# Visualization and report question (HW10.ipynb cells [28], [30], [32]; the JPEG TODO is filled in).
# Run hw10.py first: it reads the adversarial images in fgsm/.
# Saves the cell [28] figure as visualization.png and prints the titles cells [30] and [32] would plot.
# Source model resnet110_cifar10, vanilla fgsm attack on dog/dog2.png, then JPEG compression (rate 70).
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
import imgaug.augmenters as iaa

from config import device
from dataset import transform

model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
model.eval()  # in the notebook gen_adv_examples() had already put the model in eval mode

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 20))
cnt = 0
for i, cls_name in enumerate(classes):
    path = f'{cls_name}/{cls_name}1.png'
    # benign image
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./data/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    # adversarial image
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./fgsm/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
plt.tight_layout()
plt.savefig('visualization.png')

# original image
path = f'dog/dog2.png'
im = Image.open(f'./data/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
print(f'benign: dog2.png  {classes[predict]}: {prob:.2%}')

# adversarial image
im = Image.open(f'./fgsm/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
print(f'adversarial: dog2.png  {classes[predict]}: {prob:.2%}')

# Passive Defense - JPEG compression by imgaug package, compression rate set to 70
# pre-process image
x = transforms.ToTensor()(im)*255
x = x.permute(1, 2, 0).numpy()
compressed_x = x.astype(np.uint8)

# TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
# compressed_x = ...
compressed_x = iaa.JpegCompression(compression=70)(image=compressed_x)

logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
print(f'JPEG adversarial: dog2.png  {classes[predict]}: {prob:.2%}')
