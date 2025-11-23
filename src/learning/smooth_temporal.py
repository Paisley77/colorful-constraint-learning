import torch

def smooth_always_operator(manifold_distances, beta=1.0):
    """
    Differentiable Always operator: min_t distance -> smooth approximation
    """
    # Robust min via log-sum-exp
    return -torch.logsumexp(beta * manifold_distances, dim=-1) / beta

def temporal_constraint_loss(expert_hsv, violator_hsv, manifold, margin=0.5):
    """
    Loss that pushes expert trajectories to satisfy Always(near_manifold)
    and violator trajectories to violate it.
    """
    # Compute manifold distances for entire trajectories
    expert_dists = manifold(expert_hsv)  # [batch, seq_len]
    violator_dists = manifold(violator_hsv)
    
    # Apply smooth Always operator
    expert_satisfaction = smooth_always_operator(expert_dists)
    violator_satisfaction = smooth_always_operator(violator_dists)
    
    # Expert should be close to manifold (negative distance), violator far
    loss = (torch.clamp(margin - expert_satisfaction, min=0).mean() +  # Expert too far
            torch.clamp(violator_satisfaction + margin, min=0).mean())  # Violator too close
    
    return loss