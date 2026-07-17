from contextlib import contextmanager
from operator import attrgetter

import torch


class PersonaSteerer:
    def __init__(
        self,
        model,
        model_id,
        target_layers,
        ruv,
        projection=None,
        coef=1.0,
    ):
        self.model = model
        self.model_id = model_id
        self.target_layers = target_layers
        self.ruv = ruv
        self.coef = coef

        self.projection = projection

        layers_paths = {
            "Qwen/Qwen2.5-VL-7B-Instruct": "model.language_model.layers",
            "HuggingFaceM4/Idefics3-8B-Llama3": "model.model.text_model.layers",
        }

        # layers might be under model.module if using DataParallel or DistributedDataParallel
        layers_path = layers_paths[self.model_id]
        try:
            self.layers = attrgetter(layers_path)(model.module)
        except AttributeError:
            self.layers = attrgetter(layers_path)(model)

        self.activations = {}
        self.hooks = []

    def get_hook(self, layer_idx):
        def extract_hook(module, input, output):

            activation = output[0] if isinstance(output, tuple) else output

            # match sequence length
            matched_ruv = (
                self.ruv[layer_idx].unsqueeze(1).repeat(1, activation.shape[1], 1)
            )

            activation = activation + matched_ruv.to(activation.device) * self.coef


            if isinstance(output, tuple):
                output = (activation,) + output[1:]
            else:
                output = activation

            return output

        return extract_hook

    def __enter__(self):

        for idx in self.target_layers:
            self.hooks.append(
                self.layers[idx].register_forward_hook(self.get_hook(idx))
            )


    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()
class ActivationExtractor:
    def __init__(
        self, model, model_id, target_layers, detach=False, module="down_proj"
    ):
        self.model = model
        self.model_id = model_id
        self.target_layers = target_layers
        self.detach = detach
        self.module = module

        layers_paths = {
            "Qwen/Qwen2.5-VL-7B-Instruct": "model.language_model.layers",
            "HuggingFaceM4/idefics2-8b": "model.text_model.layers",
            # "HuggingFaceM4/Idefics3-8B-Llama3": "model.text_model.layers",
            "HuggingFaceM4/Idefics3-8B-Llama3": "model.model.text_model.layers"
        }

        # layers might be under model.module if using DataParallel or DistributedDataParallel
        layers_path = layers_paths[self.model_id]
        try:
            self.layers = attrgetter(layers_path)(model.module)
        except AttributeError:
            self.layers = attrgetter(layers_path)(model)

        self.activations = {}
        self.hooks = []

    def get_hook(self, layer_idx):
        def extract_hook(module, input, output):
            # The output of a decoder layer is often a tuple (hidden_state, caches, ...)
            # We are interested in the hidden_state, which is the first element.
            activation = output[0] if isinstance(output, tuple) else output
            if self.detach:
                self.activations[layer_idx] = activation.detach()
            else:
                self.activations[layer_idx] = activation

        return extract_hook

    def __enter__(self):

        for idx in self.target_layers:
            if self.module == "down_proj":
                self.hooks.append(
                    self.layers[idx].mlp.down_proj.register_forward_hook(
                        self.get_hook(idx)
                    )
                )
            if self.module == "residual":
                self.hooks.append(
                    self.layers[idx].register_forward_hook(self.get_hook(idx))
                )
            if self.module == "self_attn":
                self.hooks.append(
                    self.layers[idx].post_attention_layernorm.register_forward_hook(
                        self.get_hook(idx)
                    )
                )
        return self.activations

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()


def compute_PCA_debias(forget_acts, retain_acts, n_components,
                       return_projection=False, relaxed_labels=None):
    """
    Compute PCA-based debiased representations or return the projection matrix.

    When return_projection=False (default): returns debiased forget and retain
    activations — analogous to ``compute_inlp_debias``.

    When return_projection=True: returns the D×D projection matrix ``P_debias``
    and the training mean ``mu_train`` so that downstream code can apply the
    projection on-the-fly (e.g. inside a forward hook after steering).

    The unbiased subspace is selected as the K eigenvectors of the training-set
    covariance that have the *smallest absolute projection* of the forget mean
    (or the target×protected intersection mean when ``relaxed_labels`` is given).

    Args:
        forget_acts: tensor [N_f, D]
        retain_acts: tensor [N_r, D]
        n_components: int, number of PCA components for the unbiased subspace
        return_projection: bool, if True return (P_debias, mu_train) instead of
                          debiased activations
        relaxed_labels: 1D numpy int64 array of length N_f + N_r, optional.
                        1 = target×protected intersection, 0 = others.
                        When given, the subspace targets this split instead of
                        forget-vs-retain.

    Returns:
        If return_projection=False:
            (debiased_forget, debiased_retain) — tensors [N_f, D], [N_r, D]
        If return_projection=True:
            (P_debias, mu_train) — P_debias is [D, D], mu_train is [D]
    """
    if n_components <= 0:
        if return_projection:
            D = forget_acts.shape[1]
            return torch.eye(D, device=forget_acts.device, dtype=forget_acts.dtype), \
                   torch.zeros(D, device=forget_acts.device, dtype=forget_acts.dtype)
        return forget_acts, retain_acts

    f = forget_acts.float()
    r = retain_acts.float()
    train_activations = torch.cat([f, r], dim=0)

    D = train_activations.shape[1]
    k = min(n_components, D)

    mu_train = train_activations.mean(0)

    if relaxed_labels is not None:
        relaxed_labels_t = torch.from_numpy(relaxed_labels).to(train_activations.device)
        mu_p = train_activations[relaxed_labels_t == 1].mean(0) - mu_train
    else:
        mu_f = f.mean(0)
        mu_p = mu_f - mu_train

    standardized_train = train_activations - mu_train
    cov = torch.cov(standardized_train.T)
    _, V = torch.linalg.eig(cov)
    V = torch.real(V)

    mu_p = torch.matmul(V.T, mu_p)
    topk_unbiased = mu_p.abs().topk(k, largest=False).indices
    V_u = V[:, topk_unbiased]  # [D, K]

    if return_projection:
        P_debias = torch.matmul(V_u, V_u.T)  # [D, D]
        return P_debias, mu_train

    X_fc = f - mu_train
    X_rc = r - mu_train
    deb_f = torch.matmul(torch.matmul(X_fc, V_u), V_u.T) + mu_train
    deb_r = torch.matmul(torch.matmul(X_rc, V_u), V_u.T) + mu_train

    orig_device = forget_acts.device
    orig_dtype = forget_acts.dtype
    deb_f = deb_f.to(device=orig_device, dtype=orig_dtype)
    deb_r = deb_r.to(device=orig_device, dtype=orig_dtype)

    return deb_f, deb_r