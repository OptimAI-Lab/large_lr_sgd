import torch

import os
import random
from optimizers.sgd_ll import SGD_LL

def set_seed_deterministic(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # for reproducibility
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)

def get_trainable_parameters(model, args):
    trainable_params = []
    for name, prm in model.named_parameters():
        trainable_params.append({"params": prm, "lr": args.lr})
    return trainable_params


def build_optimizer(model, trainable_params, args):   
    if args.optimizer.lower() == "sgd_ll":
        
        id_to_name = {}
        for param_name, p in model.named_parameters():
            id_to_name[id(p)] = param_name
            
        optimizer = SGD_LL(trainable_params, lr=args.lr, id_to_name=id_to_name, momentum=args.momentum, update_limit=args.update_limit, col_thresh=args.col_thresh)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer