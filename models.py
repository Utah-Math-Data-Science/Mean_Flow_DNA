import torch.nn as nn
import torch



class MLPModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.num_cls = num_cls
        self.classifier = classifier
        self.args = args

        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU()  
        )
        
        # Input embedding handles expanded simplex if needed
        self.input_expansion = 1 if classifier and not args.cls_expanded_simplex else 2
        self.embedder = nn.Linear(self.input_expansion * alphabet_size, args.hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim + 2 * args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),  # Add this
            nn.Dropout(0.1),  
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),  # Add this
            nn.Dropout(0.1),  
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim if classifier else alphabet_size)
        )
        
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, num_cls)
            )
        
        if getattr(args, 'cls_free_guidance', False) and not classifier:
            self.cls_embedder = nn.Embedding(num_cls + 1, args.hidden_dim)

    def forward(self, x, t, r, cls=None):
        """
        Args:
            x: [batch, seq_len, alphabet_size * expansion]
            t: [batch] (current time)
            r: [batch] (previous time)
            cls: optional [batch] class labels
        """
        # Time embeddings for both t and r
        t_embed = self.time_embedder(t)  # [batch, hidden_dim]
        r_embed = self.time_embedder(r)  # [batch, hidden_dim]
        
        feat = self.embedder(x)  # [batch, seq_len, hidden_dim]
        
        feat = feat + t_embed.unsqueeze(1) + r_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim]
        
        if getattr(self.args, 'cls_free_guidance', False) and not self.classifier and cls is not None:
            cls_embed = self.cls_embedder(cls)  # [batch, hidden_dim]
            feat = feat + cls_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim]
        
        # Prepare MLP input with full context
        mlp_input = torch.cat([
            feat,
            t_embed.unsqueeze(1).expand(-1, x.size(1), -1),
            r_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        ], dim=-1)  # [batch, seq_len, hidden_dim * 3]
        
        # Process through MLP
        output = self.mlp(mlp_input)  # [batch, seq_len, output_dim]
        
        if self.classifier:
            return self.cls_head(output.mean(dim=1))  # [batch, num_cls]
        return output  # [batch, seq_len, alphabet_size]

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)