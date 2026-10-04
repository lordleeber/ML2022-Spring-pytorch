"""Fill the TODO(本機實測) marker of docs/HW09/ch05.html (review of PR #15).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch05_review.py
Replaces line 265 of explain_cnn.py with the three lines from the ch05 quiz
(by editing the source text and exec-ing it; the file is not modified), runs
integrated_gradients(), and writes the figure to ./output/ch05_quiz1.png (not used in the book).
"""
import sys
import torch

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels

SRC = open('explain_cnn.py').read()
LIB = SRC[:SRC.index('if __name__')]
old = '        # [0] to get rid of the first channel (1,3,128,128)\n        return integrated_grads[0]\n'
new = ('        # [0] to get rid of the first channel (1,3,128,128)\n'
       '        result = integrated_grads[0] * input_image[0].detach().cpu().numpy()\n'
       "        print('sum of IG:', result.sum())\n"
       '        return result\n')
assert LIB.count(old) == 1
ns = {'__name__': 'explain_variant'}
exec(compile(LIB.replace(old, new), 'explain_cnn.py', 'exec'), ns)
ns['output_dir'] = './output/'
ns['save_fig'].__globals__['output_dir'] = './output/'

model = Classifier().cuda()
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
orig_save = ns['save_fig']
ns['save_fig'] = lambda fig, name: orig_save(fig, 'ch05_quiz1.png')
ns['integrated_gradients'](model, images, labels)
