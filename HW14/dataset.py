import os

import torch.nn.functional as F
from torchvision import datasets, transforms


# Rotate MNIST to generate 10 tasks

def _rotate_image(image, angle):
  if angle is None:
    return image

  image = transforms.functional.rotate(image, angle=angle)
  return image

def get_transform(angle=None):
  transform = transforms.Compose([transforms.ToTensor(),
                   transforms.Lambda(lambda x: _rotate_image(x, angle)),
                   Pad(28)
                   ])
  return transform

class Pad(object):
  def __init__(self, size, fill=0, padding_mode='constant'):
    self.size = size
    self.fill = fill
    self.padding_mode = padding_mode
    
  def __call__(self, img):
    # If the H and W of img is not equal to desired size,
    # then pad the channel of img to desired size.
    img_size = img.size()[1]
    assert ((self.size - img_size) % 2 == 0)
    padding = (self.size - img_size) // 2
    padding = (padding, padding, padding, padding)
    return F.pad(img, padding, self.padding_mode, self.fill)

class Data():
  def __init__(self, path, train=True, angle=None):
    transform = get_transform(angle)
    self.dataset = datasets.MNIST(root=os.path.join(path, "MNIST"), transform=transform, train=train, download=True)
