import torch
import numpy as np
def save_model(policy_net, value_net, policy_optimizer, value_optimizer, path):
    checkpoint = {
        'policy_net_state_dict': policy_net.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        'value_optimizer_state_dict': value_optimizer.state_dict()
    }
    torch.save(checkpoint, path)

# def load_model(policy_net, value_net, policy_optimizer, value_optimizer, path):
#     checkpoint = torch.load(path)
#     policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
#     value_net.load_state_dict(checkpoint['value_net_state_dict'])
#     policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
#     value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
#     return policy_net, value_net, policy_optimizer, value_optimizer

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval() 
    return model

def dict_2_tensor(state_dict, args):
    size =  int(np.sqrt(args.state_dim))
   
    state = torch.zeros((size, size), dtype=torch.float32)
    state[state_dict['agent'][0], state_dict['agent'][1]] = 1
    state[state_dict['target'][0], state_dict['target'][1]] = -1
    
    return state.unsqueeze(0)