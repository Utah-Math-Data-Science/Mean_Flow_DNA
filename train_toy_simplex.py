from toy_data import ToyDataset
from DNA_module import DNAModule 

from argparse import Namespace
import os
from torch.utils.data import DataLoader
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    num_integration_steps = 10, 
    limit_val_batches=1000,
    limit_train_batches = 1000
)


trainer = pl.Trainer(
    default_root_dir="./saved_models",
    max_epochs=args.max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[
        ModelCheckpoint(
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            filename="best-{epoch}-{train_loss:.2f}"
        )
    ],
    log_every_n_steps=10,
    check_val_every_n_epoch=10,
    limit_val_batches=args.limit_val_batches,
    limit_train_batches = args.limit_train_batches
)

os.makedirs("./saved_models", exist_ok=True)

train_ds = ToyDataset(args)
val_ds = ToyDataset(args)  # Create fresh instance if possible

toy_data = train_ds        
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory = True)

model = DNAModule(args, train_ds.alphabet_size, train_ds.num_cls, toy_data)
model.to(Device)
trainer.fit(model, train_loader, val_loader)

