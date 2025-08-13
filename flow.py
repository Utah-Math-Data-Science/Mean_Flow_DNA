import torch 

def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights

def sample_conditional_path(args, seq, alphabet_size, device):
    B, L = seq.shape
    t = torch.rand(B, device = device)
    r = t * torch.rand_like(t, device = device)  # r ∈ [0,t)
    K = args.toy_simplex_dim

    # separate this out into utils at some point, not right now though
    x1 = torch.nn.functional.one_hot(seq, K).float().to(device)  # [B,L,K]
    x0 = torch.distributions.Dirichlet(torch.ones(K)).sample([B,L]).to(device) # [B,L,K]
    xt = t[:,None,None]*x1 + (1-t[:,None,None])*x0
    return x0, xt, x1, t, r
