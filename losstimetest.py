import torch
import pytorch_metric_learning as pml
import numpy as np

e = np.random.rand(256*256)
p = np.random.randint(0, 5, size=(256*256))
# labels = np.random.randint(0, 5, size=(256*256))

embeddings = torch.Tensor(e)
preds = torch.Tensor(p)
