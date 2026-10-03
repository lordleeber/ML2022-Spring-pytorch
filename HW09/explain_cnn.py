import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from skimage.segmentation import slic
from lime import lime_image
from model import Classifier
from dataset import FoodDataset, get_paths_labels


"""# Homework 9 - Explainable AI (Part 1 CNN)

Every figure is saved to output/ instead of being shown inline as in Colab.
"""

ckptpath = './checkpoint.pth'
dataset_dir = './food/'
output_dir = './output/'


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def save_fig(fig, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


"""# Lime (Q1~4)
[Lime](https://github.com/marcotcr/lime) is a package about explaining what machine learning classifiers are doing.
"""

def lime_explain(model, images, labels):
    def predict(input):
        # input: numpy array, (batches, height, width, channels)
        model.eval()
        input = torch.FloatTensor(input).permute(0, 3, 1, 2)
        # pytorch tensor, (batches, channels, height, width)
        output = model(input.cuda())
        return output.detach().cpu().numpy()

    def segmentation(input):
        # split the image into 200 pieces with the help of segmentaion from skimage
        return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)

    fig, axs = plt.subplots(1, len(images), figsize=(15, 8))
    # fix the random seed to make it reproducible
    np.random.seed(16)
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # numpy array for lime
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance

        lime_img, mask = explaination.get_image_and_mask(
            label=label.item(),
            positive_only=False,
            hide_rest=False,
            num_features=11,
            min_weight=0.05
        )
        # turn the result from explainer to the image
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.ImageExplanation.get_image_and_mask
        axs[idx].imshow(lime_img)
    save_fig(fig, 'lime.png')


"""## Saliency Map (Q5~9)
The partial derivative of the loss with respect to the input image shows how much each pixel matters.
"""

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # we want the gradient of the input x
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    # saliencies = x.grad.abs().detach().cpu()
    saliencies, _ = torch.max(x.grad.abs().detach().cpu(), dim=1)

    # We need to normalize each image, because their gradients might vary in scale
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies


def saliency(model, images, labels):
    saliencies = compute_saliency_maps(images, labels, model)

    fig, axs = plt.subplots(2, len(images), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            if row == 0:
                # pytorch image is (channels, height, width); matplotlib wants (height, width, channels)
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
            else:
                axs[row][column].imshow(img.numpy(), cmap=plt.cm.hot)
    save_fig(fig, 'saliency.png')


"""## Smooth Grad (Q10~13)
Randomly add noise to the image, compute a heatmap each time, and average them.
The average is more robust to noisy gradients.

ref: https://arxiv.org/pdf/1706.03825.pdf
"""

def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
    model.eval()

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros(x.cuda().unsqueeze(0).size())
    for i in range(epoch):
        # generate random noise (the original passes sigma**2 as the std)
        noise = x.new_empty(x.size()).normal_(mean, sigma**2)
        x_mod = (x + noise).unsqueeze(0).cuda()
        x_mod.requires_grad_()

        y_pred = model(x_mod)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pred, y.cuda().unsqueeze(0))
        loss.backward()

        # like the method in saliency map
        smooth += x_mod.grad.abs().detach().cpu().numpy()
    smooth = normalize(smooth / epoch)  # don't forget to normalize
    # smooth = smooth / epoch # try this line to answer the question
    return smooth


def smoothgrad(model, images, labels):
    smooth = []
    for i, l in zip(images, labels):
        smooth.append(smooth_grad(i, l, model, 500, 0.4))
    smooth = np.stack(smooth)

    fig, axs = plt.subplots(2, len(images), figsize=(15, 8))
    for row, target in enumerate([images, smooth]):
        for column, img in enumerate(target):
            axs[row][column].imshow(np.transpose(np.asarray(img).reshape(3, 128, 128), (1, 2, 0)))
    save_fig(fig, 'smoothgrad.png')


"""## Filter Explanation (Q14~17)
- Filter activation: pick up some images, and check which part of the image activates the filter
- Filter visualization: look for which kind of image can activate the filter the most

A forward hook grabs the output of an intermediate CNN layer without changing forward().
"""

layer_activations = None
def filter_explanation(x, model, cnnid, filterid, iteration=100, lr=1):
    # x: input image
    # cnnid: cnn layer id
    # filterid: which filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # When the model forwards through the layer[cnnid], it calls the hook,
    # which saves the output of the layer[cnnid]

    # Filter activation: x passing the filter will generate the activation map
    model(x.cuda())  # forward

    # Pick up the activation map of the filter given by filterid
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    # Filter visualization: find the image that can activate the filter the most
    x = x.cuda()
    x.requires_grad_()
    # input image gradient
    optimizer = Adam([x], lr=lr)
    # Use optimizer to modify the input image to amplify filter activation
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)

        # We want to maximize the filter activation's summation, so we add a negative sign
        objective = -layer_activations[:, filterid, :, :].sum()

        objective.backward()
        optimizer.step()
    filter_visualizations = x.detach().cpu().squeeze()

    # The hook stays registered until removed
    hook_handle.remove()

    return filter_activations, filter_visualizations


def filter_explain(model, images, cnnid):
    filter_activations, filter_visualizations = filter_explanation(images, model, cnnid=cnnid, filterid=0, iteration=100, lr=0.1)

    fig, axs = plt.subplots(3, len(images), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    # Plot filter activations
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    # Plot filter visualization
    for i, img in enumerate(filter_visualizations):
        axs[2][i].imshow(normalize(img.permute(1, 2, 0)))
    save_fig(fig, f'filter_cnn{cnnid}.png')


"""## Integrated Gradients (Q18~20)"""

class IntegratedGradients():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate scaled xbar images
        xbar_list = [input_image * step / steps for step in range(steps)]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # We want to get the gradients of the input image
        input_image.requires_grad = True
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.zeros(1, model_output.size()[-1]).cuda()
        one_hot_output[0][target_class] = 1
        # Backward
        model_output.backward(gradient=one_hot_output)
        self.gradients = input_image.grad
        # [0] to get rid of the first channel (1,3,128,128)
        gradients_as_arr = self.gradients.cpu().numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an image composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad / steps
        # [0] to get rid of the first channel (1,3,128,128)
        return integrated_grads[0]


def integrated_gradients(model, images, labels):
    images = images.cuda()

    IG = IntegratedGradients(model)
    integrated_grads = []
    for i, img in enumerate(images):
        img = img.unsqueeze(0)
        integrated_grads.append(IG.generate_integrated_gradients(img, labels[i], 10))
    fig, axs = plt.subplots(2, len(images), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.cpu().permute(1, 2, 0))
    for i, img in enumerate(integrated_grads):
        axs[1][i].imshow(np.moveaxis(normalize(img), 0, -1))
    save_fig(fig, 'integrated_gradients.png')


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    model = Classifier().cuda()
    checkpoint = torch.load(ckptpath)
    print(model.load_state_dict(checkpoint['model_state_dict']))
    # It should display: <All keys matched successfully>

    train_paths, train_labels = get_paths_labels(dataset_dir)
    train_set = FoodDataset(train_paths, train_labels, mode='eval')

    # The images for observation, marked from 0 to 9. They should be the same as those in the slides.
    # 11 categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit
    img_indices = [i for i in range(10)]
    images, labels = train_set.getbatch(img_indices)
    print('labels:', labels.tolist())

    fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[i].imshow(img.cpu().permute(1, 2, 0))
    save_fig(fig, 'images.png')

    lime_explain(model, images, labels)
    saliency(model, images, labels)
    smoothgrad(model, images, labels)
    filter_explain(model, images, cnnid=6)
    filter_explain(model, images, cnnid=23)
    integrated_gradients(model, images, labels)
