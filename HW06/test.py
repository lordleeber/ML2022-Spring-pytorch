from trainer_gan import TrainerGAN
from config import *

"""# Inference
In this section, we will use trainer to train model

## Inference through trainer
"""

# save the 1000 images into ./output folder
trainer = TrainerGAN(config)
import glob, os
G_path = max(glob.glob(f'{workspace_dir}/checkpoints/*_GAN/G_*.pth'), key=os.path.getmtime)  # latest generator checkpoint
print(f'Inference with {G_path}')
trainer.inference(G_path)

"""## Prepare .tar file for submission"""

# Commented out IPython magic to ensure Python compatibility.
# %cd output
# !tar -zcf ../submission.tgz *.jpg
# %cd ..
