# HW13 Network Compression: configs (HW13.ipynb cell [5])
cfg = {
    'dataset_root': './food11-hw13',
    'save_dir': './outputs',
    'exp_name': "simple_baseline",
    'batch_size': 64,
    'lr': 3e-4,
    'seed': 20220013,
    'loss_fn_type': 'CE', # simple baseline: CE, medium baseline: KD. See the Knowledge_Distillation part for more information.
    'weight_decay': 1e-5,
    'grad_norm_max': 10,
    'n_epochs': 10, # train more steps to pass the medium baseline.
    'patience': 300,
}
