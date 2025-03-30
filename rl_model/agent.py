import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import tqdm
import time
import pickle, os

CUDA_device = 0
eps = 1e-12

def soft_cross_entropy(
    pred, target_indices, target_probas, weight=None, reduction="mean", ignore_index=-1
):
    """Computes the cross entropy loss using soft labels.

    Here the soft labels are defined through the parameters `target_indices`
    and `target_probas`. They respectively represent the class indices involved
    in the target distribution and their corresponding probability.
    The provided `ignore_index` can be used as padding element in the `target_indices`
    field.

    Per definition, we have (https://en.wikipedia.org/wiki/Cross_entropy):
        CE(p,q) = -(p * log(q)).sum()
    With a provided weight per class, the computation becomes:
        CE(p,q,w) = -(p * log(q)).sum() * (p * w).sum()

    Parameters
    ----------
    pred: tensor
        a tensor of size `N x C x *` where N is the batch size, C is the number
        of classes, and `*` represents any other dimensions. This tensor represents
        the logit values.
    target_indices: tensor
        a tensor of size `N x D x *` where N is the batch size, D <= C is the number
        of classes present in the soft distribution, and `*` represents
        any other dimensions. It must match the tailing dimensions of `pred`.
    target_probas: tensor
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    weight: tensor
        a manual rescaling weight given to each class. It is a 1-D tensor of size
        `C`. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the gradient. Default: -1

    Return
    ----------
    result: tensor
        the computed loss.

    """
    assert reduction in ["none", "mean", "sum"]

    dim = pred.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    dim = target_indices.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    assert (weight is None) or (weight.dim() == 1 and weight.size(0) == pred.size(1))

    if pred.size(0) != target_indices.size(0):
        raise ValueError(
            f"Expected input batch_size ({pred.size(0)}) to match "
            f"target batch_size ({target_indices.size(0)})."
        )

    if pred.size(1) < target_indices.size(1):
        raise ValueError(
            f"Expected input class_size ({pred.size(1)}) to be greater/equal"
            f"than target class_size ({target_indices.size(1)})."
        )
    if target_indices.size()[2:] != pred.size()[2:]:
        out_size = target_indices.size()[:2] + pred.size()[2:]
        raise ValueError(
            f"Expected target_indices size {out_size} (got {target_indices.size()})"
        )
    if target_indices.size() != target_probas.size():
        raise ValueError(
            f"Expected target_probas size {target_indices.size()} "
            f"(got {target_probas.size()})"
        )

    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    mask = target_indices != ignore_index
    masked_indices = target_indices * mask
    tmp_weight = 1.0 if weight is None else weight[masked_indices]
    avg_log_probs = (mask * log_probs.gather(1, masked_indices) * target_probas).sum(
        dim=1
    )
    avg_weight = (
        1.0 if weight is None else (tmp_weight * mask * target_probas).sum(dim=1)
    )
    result = -(avg_weight * avg_log_probs)

    if reduction == "sum":
        result = result.sum()
    elif reduction == "mean":
        result = result.mean() if weight is None else result.sum() / avg_weight.sum()

    return result



class sym_acquire_func(nn.Module):
    """docstring for Net"""
    def __init__(self, state_size, action_size):
        super(sym_acquire_func, self).__init__()
        
        # Adapt first layer to match state_size (could be 241 or 922)
        self.fc1 = nn.Linear(state_size, 1024*2)
        self.fc2 = nn.Linear(1024*2, 2048*1)
        self.fc3 = nn.Linear(2048*1, 1024*2)
        self.out = nn.Linear(1024*2, action_size)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        action_prob = F.softmax(self.out(x), dim = 1)

        return action_prob

class diagnosis_func(nn.Module):
    """docstring for Net"""
    def __init__(self, state_size, disease_size):
        super(diagnosis_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024*2)
        self.fc2 = nn.Linear(1024*2, 1024*2)
        self.out = nn.Linear(1024*2, disease_size)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.out(x)
        return output


class Policy_Gradient_pair_model(object):
    def __init__(self, state_size, disease_size, symptom_size, LR = 1e-4, Gamma = 0.99, Eta = 0.01):
        self.original_state_size = state_size  # Store the original state size (241)
        
        # For loading the model, we'll use the state size from the trained model (922)
        self.trained_state_size = 922
        
        # Create models with the trained state size for loading weights
        self.policy = sym_acquire_func(self.trained_state_size, symptom_size)
        self.classifier = diagnosis_func(self.trained_state_size, disease_size)
        
        self.lr = LR
        
        # Check if CUDA is available
        self.device = torch.device("cuda:" + str(CUDA_device) if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.classifier.to(self.device)
        
        self.optimizer_p = torch.optim.Adam(self.policy.parameters(), lr=LR/5)
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=LR)
        self.counter = 1
        self.cross_entropy = nn.CrossEntropyLoss()
        # hyper_params
        self.gamma = Gamma
        self.eta = Eta
        
    def convert_state(self, state):
        """Convert from original state size (241) to trained state size (922)."""
        # If state is already the right size, return it
        if state.shape[-1] == self.trained_state_size:
            return state
            
        # Otherwise, pad with zeros to match the trained state size
        if len(state.shape) == 1:
            # Single state vector
            expanded_state = np.zeros(self.trained_state_size, dtype=np.float32)
            expanded_state[:min(state.shape[0], self.trained_state_size)] = state[:min(state.shape[0], self.trained_state_size)]
            return expanded_state
        else:
            # Batch of state vectors
            batch_size = state.shape[0]
            expanded_states = np.zeros((batch_size, self.trained_state_size), dtype=np.float32)
            expanded_states[:, :min(state.shape[1], self.trained_state_size)] = state[:, :min(state.shape[1], self.trained_state_size)]
            return expanded_states

    def create_batch(self, states, rewards_s, action_s, true_d, true_diff_ind, true_diff_prob):
        
        cumulate_R_s = []
        R_s = 0
        for r_s in rewards_s[::-1]:
            R_s = r_s + self.gamma * R_s
            cumulate_R_s.insert(0, R_s)
        
        rewards_s = np.array(rewards_s)
        ave_rewards_s = np.mean(np.sum(rewards_s, axis = 0))

        cumulate_R_s = np.array(cumulate_R_s).T
        states = np.array(states).swapaxes(0, 1)
        action_s = np.array(action_s).T
        true_d = np.array(true_d).T
        true_diff_ind = (
            None if true_diff_ind[0] is None
            else np.array(true_diff_ind).swapaxes(0, 1)
        )
        true_diff_prob = (
            None if true_diff_prob[0] is None
            else np.array(true_diff_prob).swapaxes(0, 1)
        )

        valid_sample = (cumulate_R_s != 0)

        self.batch_rewards_s = torch.from_numpy(cumulate_R_s[valid_sample]).float()
        self.batch_states = torch.from_numpy(states[valid_sample]).float()
        self.batch_action_s = torch.from_numpy(action_s[valid_sample])
        self.batch_true_d = torch.from_numpy(true_d[valid_sample])
        self.batch_true_diff_ind = (
            None if true_diff_ind is None else torch.from_numpy(true_diff_ind[valid_sample])
        )
        self.batch_true_diff_prob = (
            None if true_diff_prob is None else torch.from_numpy(true_diff_prob[valid_sample])
        )

        return valid_sample, len(self.batch_rewards_s), ave_rewards_s
    
    @torch.no_grad() 
    def choose_action_s(self, state, deterministic=False):

        self.policy.eval()
        state = torch.from_numpy(self.convert_state(state)).float()
        probs = self.policy.forward(state.to(self.device))
        m = Categorical(probs)
        if not deterministic:
            action = m.sample().detach().cpu().squeeze().numpy()
        else:
            action = torch.max(probs, dim=1)[1].detach().cpu().squeeze().numpy()
            
        # Handle the case when action is a scalar (0-dim array)
        if np.isscalar(action) or (isinstance(action, np.ndarray) and action.ndim == 0):
            action = np.array([int(action)])
            
        return action
    
    @torch.no_grad()
    def choose_diagnosis(self, state):

        self.classifier.eval()
        state = torch.from_numpy(self.convert_state(state)).float()
        output = self.classifier.forward(state.to(self.device)).detach().cpu().squeeze()
        
        # Handle the case when output is a single vector (not a batch)
        if output.dim() == 1:
            # Add batch dimension for processing
            output = output.unsqueeze(0)
            
        # Get diagnosis and probabilities
        diagnosis = torch.max(output, dim=1)[1].numpy()
        probabilities = torch.softmax(output, dim=1).numpy()
        
        # Handle scalar output case
        if diagnosis.ndim == 0:
            diagnosis = np.array([int(diagnosis)])
        
        return diagnosis, probabilities
    
    def update_param_rl(self):  

        self.policy.train() 
        self.optimizer_p.zero_grad()
        # Convert batch states to trained model size
        converted_states = torch.from_numpy(self.convert_state(self.batch_states.cpu().numpy())).float().to(self.device)
        reward_tensor = self.batch_rewards_s.to(self.device)
        action_s_tensor = self.batch_action_s.to(self.device)
        prob_tensor = self.policy.forward(converted_states)
        #Policy Loss
        m = Categorical(prob_tensor)
        log_prob_tensor = m.log_prob(action_s_tensor)
        policy_loss = - (log_prob_tensor * (reward_tensor)).mean()
        #entropy Loss
        entropy_loss = - torch.max(torch.tensor([self.eta-self.counter*0.00001, 0])) * m.entropy().mean()
        loss = policy_loss + entropy_loss
        loss.backward()
        self.optimizer_p.step()

        self.counter += 1

    def update_param_c(self):

        self.classifier.train()
        self.optimizer_c.zero_grad()
        # Convert batch states to trained model size
        converted_states = torch.from_numpy(self.convert_state(self.batch_states.cpu().numpy())).float().to(self.device)
        label_tensor = self.batch_true_d.to(self.device)
        diff_ind_tensor = (
            None if self.batch_true_diff_ind is None
            else self.batch_true_diff_ind.to(self.device)
        )
        diff_prob_tensor = (
            None if self.batch_true_diff_prob is None
            else self.batch_true_diff_prob.to(self.device)
        )
        output_tensor = self.classifier.forward(converted_states)
        if diff_ind_tensor is None or diff_prob_tensor is None:
            loss = self.cross_entropy(output_tensor, label_tensor)
        else:
            loss = soft_cross_entropy(output_tensor, diff_ind_tensor, diff_prob_tensor)
        loss.backward()
        self.optimizer_c.step()
    
    def change_lr(self):
        self.lr = self.lr /2 
        if self.lr < 1e-5:
            self.lr = 1e-4
        for param_group in self.optimizer_p.param_groups:
            param_group['lr'] = self.lr/2
        for param_group in self.optimizer_c.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_m.param_groups:
            param_group['lr'] = self.lr/2
  
    def save_model(self, args, prefix=""):
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        torch.save(self.policy.state_dict(), os.path.join(args.save_dir, f"{prefix}policy_{info}.pth"))
        torch.save(self.classifier.state_dict(), os.path.join(args.save_dir, f"{prefix}classifier_{info}.pth"))
       
    def load_model(self, args, prefix=""):
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        self.policy.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"{prefix}policy_{info}.pth"), map_location=self.device))
        self.classifier.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"{prefix}classifier_{info}.pth"), map_location=self.device))


    def train(self):
        self.policy.train()
        self.classifier.train()

    def eval(self):
        self.policy.eval()
        self.classifier.eval()
        
    def predict(self, answers):
        """Compatibility method for QuestionSelector.
        
        Args:
            answers: Dictionary of evidence code to answer value
            
        Returns:
            Dict: Mapping of condition indices to probabilities
        """
        # Convert answers to state vector (we need to implement this)
        # For now, return a uniform distribution over all diseases
        import numpy as np
        disease_size = self.classifier.out.out_features
        probs = np.ones(disease_size) / disease_size
        return {i: prob for i, prob in enumerate(probs)}
        
    def predict_d_prob(self, state, temperature=None):
        """This is a helper function to directly get the diagnosis probabilities."""
        self.classifier.eval()
        # Convert state to trained model size
        converted_state = torch.from_numpy(self.convert_state(np.array([state]))).float().to(self.device)
        
        with torch.no_grad():
            out = self.classifier.forward(converted_state)
            if temperature is not None:
                # Apply temperature scaling
                out = out / temperature
            probs = F.softmax(out, dim=1)
        
        return probs[0].cpu().numpy()
        