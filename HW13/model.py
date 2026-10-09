# Student and teacher models (HW13.ipynb cells [16], [19], [22])
import os

import torch
import torch.nn as nn
import torchvision

# Example implementation of Depthwise and Pointwise Convolution
def dwpw_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels), #depthwise convolution
        nn.Conv2d(in_channels, out_channels, 1), # pointwise convolution
    )

# Define your student network here. You have to copy-paste this code block to HW13 GradeScope before deadline.
# We will use your student network definition to evaluate your results(including the total parameter amount).

class StudentNet(nn.Module):
    def __init__(self):
      super().__init__()

      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(64, 100, 3),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )

    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)

def get_student_model(): # This function should have no arguments so that we can get your student network by directly calling it.
    # you can modify or do anything here, just remember to return an nn.Module as your student network.
    return StudentNet()

# End of definition of your student model and the get_student_model API
# Please copy-paste the whole code block, including the get_student_model function.


def get_teacher_model(dataset_root):
    # Load provided teacher model (model architecture: resnet18, num_classes=11, test-acc ~= 89.9%)
    # The notebook uses torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11),
    # which downloads torchvision v0.10.0 from GitHub; the installed torchvision builds the same resnet18.
    teacher_model = torchvision.models.resnet18(weights=None, num_classes=11)
    # load state dict
    teacher_ckpt_path = os.path.join(dataset_root, "resnet18_teacher.ckpt")
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))
    # Now you already know the teacher model's architecture. You can take advantage of it if you want to pass the strong or boss baseline.
    # Source code of resnet in pytorch: (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
    # You can also see the summary of teacher model. There are 11,182,155 parameters totally in the teacher model
    # summary(teacher_model, (3, 224, 224), device='cpu')
    return teacher_model
