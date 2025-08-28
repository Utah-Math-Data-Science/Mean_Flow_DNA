import torch
import torch.nn as nn
import torch.nn.functional as F

class Dequantizer(torch.nn.Module):

    def __init__(self, K: int, learn_logits: bool = True, 
                 per_pos_alpha: bool = True, 
                 tau_init: float = 1.0):
        super().__init__()
        self.K = K
        self.learn_logits = learn_logits
        self.per_pos_alpha = per_pos_alpha
        self.tau = tau_init

        if learn_logits:
            self.logits_head = nn.Sequential(
                nn.Linear(K, K), 
                nn.ReLU(inplace=True), 
                nn.Linear(K, K)
            )

        # α in [0.5, 1): α = 0.5 + 0.5*sigmoid(a)
        if per_pos_alpha:
            # learn α per class (broadcast to [B,S,K])
            self.alpha_param = nn.Parameter(torch.zeros(1, 1, K))
        else:
            # single global α
            self.alpha_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_onehot: torch.Tensor, hard_sample: bool = False, mask: torch.Tensor = None):


        B, S, K = x_onehot.shape

        # produce logits for a learnable/sampleable distribution p
        if self.learn_logits:
            logits = self.logits_head(x_onehot)         # [B,S,K]
        else:
            logits = x_onehot.clamp_min(1e-9).log()     # treat input as probs

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))

        # sample/relax with Gumbel-Softmax (or set hard_sample=True for ST)
        p = F.gumbel_softmax(logits, tau=self.tau, hard=hard_sample, dim=-1)  # [B,S,K]

        # α lower-bounded by 0.5 ensures the argmax is preserved
        alpha = 0.5 + 0.5 * torch.sigmoid(self.alpha_param)                   # scalar or [1,1,K]
        if alpha.ndim == 3:
            alpha = alpha.expand(B, S, K)

        z = alpha * x_onehot + (1.0 - alpha) * p                              # [B,S,K]

        assert torch.argmax(z, dim=-1).equal(torch.argmax(x_onehot, dim=-1))

        return z