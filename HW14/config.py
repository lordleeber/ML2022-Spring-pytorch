# Training arguments (HW14.ipynb, "Dataloaders and Arguments")
class Args:
  task_number = 5
  epochs_per_task = 10
  lr = 1.0e-4
  batch_size = 128
  test_size=8192

args=Args()

# generate rotations for the tasks.

# generate rotated MNIST data from 10 different rotations.

angle_list = [20 * x for x in range(args.task_number)]
