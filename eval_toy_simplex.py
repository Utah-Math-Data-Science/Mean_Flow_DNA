import pytorch_lightning as pl
from torch.serialization import safe_globals

from models import MLPModel
import torch 
from argparse import Namespace
from datasets import ToyDataset
# not using argparse for now
args = Namespace(
    toy_num_cls=1,
    toy_seq_len=4,
    toy_simplex_dim=40, 
    hidden_dim = 512,
    cls_ckpt=None,
    ckpt_iterations = None,
    batch_size = 32 ,    
    num_workers = 4  , 
    prior_pseudocount = 2,    
    shuffle = False,        
    max_epochs = 5000, 
    expand_simplex = True,
    cls_free_guidance = False,
    binary_guidance = False, 
    num_integration_steps = 10, 
    limit_val_batches=1000
)
alphabet_size = 40
num_cls = 1




model = MLPModel(args, alphabet_size, num_cls, classifier = False)

required_classes = [
    Namespace,
    ToyDataset, 
]

with safe_globals(required_classes):
    checkpoint = torch.load(
        'saved_models/lightning_logs/version_120/checkpoints/best-epoch=249-train_loss=5.95.ckpt',
        weights_only=True
    )

# Remove 'model.' prefix from all keys
state_dict = checkpoint['state_dict']

new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)

