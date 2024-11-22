import numpy as np
from torch.utils.data import random_split, TensorDataset, Subset

def train_val_test_split(dataset: TensorDataset, train_val_test_ratio: tuple = (0.8, 0.1, 0.1)) -> tuple[Subset, Subset, Subset]:
  """Split dataset into train, validation, and test sets.

  Args:

      dataset (TensorDataset): dataset to split
      train_val_test_ratio (tuple): ratios of train, validation, and test sets

  Returns:
  
        tuple[Subset, Subset, Subset]: train, validation

  """
  # get size of each dataset
  train_ratio, val_ratio, test_ratio = train_val_test_ratio       
  train_set_size = int(len(dataset) * train_ratio)                
  valid_set_size = int(len(dataset) * val_ratio)                  
  test_set_size = len(dataset) - train_set_size - valid_set_size

  return random_split(dataset, [train_set_size, valid_set_size, test_set_size])