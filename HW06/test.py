from trainer_gan import TrainerGAN
from config import *

"""# Inference
In this section, we will use trainer to train model

## Inference through trainer
"""

# save the 1000 images into ./output folder
trainer = TrainerGAN(config)
trainer.inference(f'{workspace_dir}/checkpoints/2022-09-29_16-12-52_GAN/G_9.pth') # you have to modify the path when running this line

"""## Prepare .tar file for submission"""

# Commented out IPython magic to ensure Python compatibility.
# %cd output
# !tar -zcf ../submission.tgz *.jpg
# %cd ..
