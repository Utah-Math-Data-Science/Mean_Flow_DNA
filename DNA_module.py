from models import MLPModel
from flow import expand_simplex, sample_conditional_path
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict

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
        self.val_outputs = defaultdict(list)

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
        
        primal = (xt, t, r)
        tangent = (torch.zeros_like(xt), torch.ones_like(t), torch.zeros_like(r))
        u, du_dt = torch.func.jvp(model_fn, primal, tangent)
        
        u_target = v + (t - r)[:,None,None] * du_dt.detach()

        #huber seems to give better results? 
        loss = torch.nn.functional.huber_loss(u, u_target) 

        if self.stage == "val":
            logits = self.mean_flow_inference(seq)
            predicted_sequence = torch.argmax(logits, dim=-1)
            
            self.val_outputs['seqs'].append(predicted_sequence.cpu())
            
            # Compute KL divergence for THIS BATCH
            batch_one_hot = torch.nn.functional.one_hot(predicted_sequence, num_classes=self.args.toy_simplex_dim)
            batch_empirical_dist = batch_one_hot.float().mean(dim=0)  # Average over batch
            
            # Ensure no zeros for numerical stability
            eps = 1e-10
            batch_empirical_dist = batch_empirical_dist.clamp(min=eps)
            true_probs = self.toy_data.probs[0].clamp(min=eps).to(batch_empirical_dist.device)
            
            # KL(true || model) and KL(model || true)
            kl = (batch_empirical_dist * (torch.log(batch_empirical_dist) - torch.log(true_probs))).sum()
            rkl = (true_probs * (torch.log(true_probs) - torch.log(batch_empirical_dist))).sum()
            sanity_self_kl = (batch_empirical_dist * (torch.log(batch_empirical_dist) - torch.log(batch_empirical_dist))).sum(-1).mean()

            # Log batch-level metrics
            self.log("self_rkl", sanity_self_kl, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_kl", kl, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_rkl", rkl, on_step=True, on_epoch=True)

            if self.args.cls_ckpt is not None:
                self.run_cls_model(predicted_sequence, cls, clean_data=False, postfix='_generated')
        
        return loss

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.step(batch)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)



    def mean_flow_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        
        zt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        
        for n in range(self.args.num_integration_steps - 1):
            r = t_span[n]      
            t = t_span[n+1]    
            
            zt_inp, _ = expand_simplex(zt, r.expand(B), self.args.prior_pseudocount)
            u = self.model(zt_inp, r.expand(B), t.expand(B))
            
            dt = t - r
            zt = zt + dt * u
        
        return zt


    def load_model(self, checkpoint=None):
        self.model = MLPModel(self.args, self.alphabet_size, self.num_cls, classifier = False)

    def check_accuracy(self, seq_pred, original_seq):
        print(f'accuracy: {seq_pred.eq(original_seq).float().mean()}')
