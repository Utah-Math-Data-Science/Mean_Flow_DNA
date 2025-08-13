import torch.nn as nn
import torch



class MLPModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier):
        super().__init__()
        self.args = args
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls

        # Time embedding
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        
        # Input embedding - handles both regular and expanded simplex cases
        embed_input_dim = (2 if args.expand_simplex else 1) * alphabet_size
        self.embedder = nn.Linear(embed_input_dim, args.hidden_dim)
        
        # Main MLP
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim if classifier else alphabet_size)
        )
        
        # Classifier head (if needed)
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, num_cls)
            )
        
        # Classifier-free guidance (if needed)
        if getattr(args, 'cls_free_guidance', False) and not classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=num_cls + 1, 
                                           embedding_dim=args.hidden_dim)

    def forward(self, x, t, cls=None):
        """
        Args:
            x: input tensor of shape [batch, seq_len, alphabet_size] or [batch, seq_len, 2*alphabet_size]
            t: time tensor of shape [batch]
            cls: optional class tensor of shape [batch]
        """
        # Time embedding
        time_embed = self.time_embedder(t)  # [batch, hidden_dim]
        
        # Process input through embedder
        feat = self.embedder(x)  # [batch, seq_len, hidden_dim]
        
        # Add time embedding (broadcasted across sequence)
        feat = feat + time_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim]
        
        # Add class embedding if using classifier-free guidance
        if getattr(self.args, 'cls_free_guidance', False) and not self.classifier and cls is not None:
            cls_embed = self.cls_embedder(cls)  # [batch, hidden_dim]
            feat = feat + cls_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim]
        
        # Process through MLP
        feat = self.mlp(feat)  # [batch, seq_len, hidden_dim or alphabet_size]
        
        # If classifier, average over sequence and pass through head
        if self.classifier:
            return self.cls_head(feat.mean(dim=1))  # [batch, num_cls]
        return feat  # [batch, seq_len, alphabet_size]


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