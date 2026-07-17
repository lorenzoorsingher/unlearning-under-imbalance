import math
from operator import attrgetter

import torch
import torch.nn.functional as F
from tqdm import tqdm

_LAYER_PATHS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "model.language_model.layers",
    "HuggingFaceM4/Idefics3-8B-Llama3": "base_model.model.model.text_model.layers",
}



class MINE(torch.nn.Module):
    def __init__(self, z_size: int, c_size: int, hidden_size: int = 512):
        super().__init__()
        self.c_size = c_size
        input_size = z_size + c_size
        self.neural_estimator = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, z, c):
        c = F.one_hot(c, num_classes=self.c_size)
        z_c = torch.cat((z, c), dim=-1)
        return self.neural_estimator(z_c)


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + 1e-18) / input.shape[0]
        )
        return grad, None


class MutualInformation(torch.nn.Module):
    def __init__(self, mine: MINE, alpha=0.01):
        super().__init__()
        self.mine = mine
        self.alpha = alpha
        self.running_mean = None

    def forward(self, z, c, c_tilde):
        joint = self.mine(z, c).mean()
        x = self.mine(z, c_tilde)

        t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.size(0))).detach()

        if self.running_mean is None:
            self.running_mean = t_exp
        else:
            self.running_mean = (
                self.alpha * t_exp + (1 - self.alpha) * self.running_mean.item()
            )
        t_log = EMALoss.apply(x, self.running_mean)

        mi = joint - t_log
        return mi

    def train(self, mode: bool = True):
        super().train(mode)
        self.mine.train(mode)

    def eval(self):
        super().eval()
        self.mine.eval()


def get_groups(dataset, batch, ids, groups_info):
    target_attr, protected_attr = groups_info[0]
    groups_map = groups_info[1]
    num_groups = get_num_groups(groups_info)

    groups = []
    for user_id in ids:
        attrs = dataset.data[user_id]["bio"]
        if target_attr not in attrs or protected_attr not in attrs:
            # assign sample to extra group
            groups.append(num_groups - 1)
        else:
            target = attrs[target_attr]
            protected = attrs[protected_attr]
            groups.append(groups_map[f"{target}-{protected}"])
    groups = torch.tensor(groups, dtype=torch.long, device="cuda")
    return groups


def get_num_groups(groups_info):
    """
    Return num groups + 1, where the additional group is used for those samples
    that cannot be assigned to any group (e.g., children do not have political orientation)
    """
    return len(groups_info[1]) + 1


def tune_mine(model, mi, train_loader, groups_info, optim, iterations, accelerator, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    # register forward hook to store features
    def last_layer_hook(module, input, output):
        """
        output: (batch_size, hidden_dim)
        """
        hook_outputs["hidden_states"] = output[0].mean(1)

    layers_path = _LAYER_PATHS.get(model_id, _LAYER_PATHS["Qwen/Qwen2.5-VL-7B-Instruct"])
    try:
        layers = attrgetter(layers_path)(model.module)
    except AttributeError:
        layers = attrgetter(layers_path)(model)
    n_layers = len(layers)
    last_block = layers[n_layers - 1]
    hook_handle = last_block.register_forward_hook(last_layer_hook)

    # get mi and optimizer for original and unlearning model
    mi_unlearning = mi["unlearning"]
    optim_unlearning = optim["unlearning"]

    model.train()
    mi_unlearning.train()


    optim_unlearning.zero_grad()


    num_groups = get_num_groups(groups_info)
    i = -1
    bar = tqdm(range(iterations), leave=False, dynamic_ncols=True, desc="Tuning MINE")
    for _ in range(iterations // len(train_loader) + 1):
        for data in train_loader:
            i += 1
            batch, _, _, ids = data
            hook_outputs = {}

            # compute group and random group (for marginalization)
            group = get_groups(train_loader.dataset, batch, ids, groups_info)
            group_tilde = torch.randint_like(group, num_groups).to("cuda")

            with torch.no_grad():
                _ = model(**batch)

            features = hook_outputs["hidden_states"]
            features = hook_outputs["hidden_states"].float()
            features = torch.nn.functional.normalize(features, dim=-1)

            # maximize mutual information
            loss = -mi_unlearning(features, group, group_tilde)

            if torch.isnan(loss):
                print(f"MINE loss is {loss.item()}.")
                exit()

            accelerator.backward(loss)
            optim_unlearning.step()
            optim_unlearning.zero_grad()

            accelerator.log({"mine_loss": loss.item()})

            # if accelerator.is_main_process:
            #     breakpoint()

            bar.update(1)
            if i == iterations:
                break

        if i == iterations:
            break

    hook_handle.remove()


def grad_norm(module):
    params = [p for p in module.parameters() if p.grad is not None]
    norm = sum([p.grad.data.norm(2).item() ** 2 for p in params]) ** 0.5
    return norm


def gradient_clipping(model, mine):
    model_norm = grad_norm(model)
    mine_norm = grad_norm(mine)

    min_norm = min(model_norm, mine_norm)
    params = [p for p in mine.parameters() if p.grad is not None]

    for p in params:
        p.grad.data *= min_norm / (mine_norm + 1e-18)
