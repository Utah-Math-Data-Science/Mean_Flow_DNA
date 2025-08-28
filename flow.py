import torch 


def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights

def sample_conditional_path(args, seq, alphabet_size, device, dequantizer=None):
    B, L = seq.shape
    t = torch.rand(B, device=device)
    step_size = 0.1
    r = (t - step_size).clamp(min=0)
    K = args.toy_simplex_dim

    x1 = torch.nn.functional.one_hot(seq, K).float().to(device)

    if args.flow_type == 'argmax':
        x1 = dequantizer(x1)
        x0 = torch.rand_like(x1).to(device)
    else:
        x0 = torch.distributions.Dirichlet(torch.ones(K)).sample([B,L]).to(device)
        
    xt = t[:,None,None]*x1 + (1-t[:,None,None])*x0
        
    return x0, xt, x1, t, r
