from models import MLPModel
from flow import expand_simplex, sample_conditional_path
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class DNAModule(pl.LightningModule):
    def __init__(self, args, alphabet_size, num_cls, toy_data):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.alphabet_size = alphabet_size
        self.num_cls = num_cls
        self.toy_data = toy_data
        self.load_model()
        self.automatic_optimization = True

    def step(self, batch):
        seq, cls = batch
        B = seq.size(0)
        
        # 1. Sample conditional path
        x0, xt, x1, t, r = sample_conditional_path(self.args, seq, self.alphabet_size, self.device)
        
        #project the interpolated sample onto the simplex 
        xt, prior_weights = expand_simplex(xt,t, self.args.prior_pseudocount)


        # 2. Compute velocity field
        v = x1 - x0
        
        # 3. Define JVP function
        def model_fn(xt, t, r):
            return self.model(xt, t, r)
        
        # 4. Compute Jacobian-vector product
        primal = (xt, t, r)
        tangent = (torch.zeros_like(xt), torch.ones_like(t), torch.zeros_like(r))
        u, du_dt = torch.func.jvp(model_fn, primal, tangent)
        
        # 5. Mean flow target
        u_target = v + (t - r)[:,None,None] * du_dt.detach()
                # Validation-specific computations
        if self.stage == "val":
            pred_seq = self.mean_flow_inference(seq)
            accuracy = pred_seq.argmax(-1).eq(seq).float().mean()
            self.log("val_accuracy", accuracy)
            
            if self.args.cls_ckpt is not None:
                self.run_cls_model(pred_seq, cls, clean_data=False, postfix='_generated')
        
        return torch.nn.functional.mse_loss(u, u_target)

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)


    def mean_flow_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        
        # Initialize from uniform Dirichlet distribution
        zt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        
        # Time steps (reverse flow: t=1 → t=0)
        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        
        for t, r in zip(t_span[:-1], t_span[1:]):  # t > r
            # Prepare inputs with correct dimensions
            t_batch = t.expand(B)  # Shape [B]
            r_batch = r.expand(B)  # Shape [B]
            
            zt_inp, _ = expand_simplex(zt, t_batch, self.args.prior_pseudocount)
 
            
            # Compute the flow field
            u = self.model(zt_inp, t_batch, r_batch)
            
            # Mean flow update equation
            zr = zt - (t - r) * u
            
            # Project back to simplex
            zt = torch.softmax(zr, dim=-1)
            
        return zt


    def load_model(self):
        self.model = MLPModel(self.args, self.alphabet_size, self.num_cls, classifier = False)
    
    def load_classifiers(self, load_cls, load_clean_cls, requires_grad = False):
                self.cls_model = MLPModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)

