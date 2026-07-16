import torch


class SGD_LL(torch.optim.Optimizer):
    """
     SGD-LL
    """

    def __init__(
        self,
        params,
        lr=None,
        id_to_name=None,
        momentum=0.9,
        update_limit=5e-4,
        col_thresh=1e-3,
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
        )

        self.col_thresh = col_thresh
        self.update_limit = update_limit  
        self.id_to_name = id_to_name

        super().__init__(params, defaults)


    def step(self, closure=None, weight_norm=None):
        """Perform a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
                
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                
                if momentum > 0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(g)
                    buf = state["momentum"]
                    buf.lerp_(g, 1 - momentum)
                    g = buf

                # token-class-wise LM-Head grad clipping
                if "lm_head" in self.id_to_name[id(p)] and self.col_thresh != -1 :
                    l2_col_norms = torch.norm(g, p=2, dim=1, keepdim=True).clamp_min_(1e-8)
                    # compute clipping factors (<= 1 if norm > thresh, else = 1)
                    col_thresh = self.col_thresh
                    scales = (col_thresh / l2_col_norms).clamp_max_(1.0)
                    # apply scaling
                    g  = g * scales

                # layer-wise RMS grad .clipping
                norm = g.pow(2).mean().sqrt().clamp(min=1e-9)
                update_limit = self.update_limit
                if norm*lr > update_limit:
                    g =  g*(update_limit  / (norm*lr) )
    
                # apply update
                p.data.add_(g, alpha=-lr)
                
        return loss