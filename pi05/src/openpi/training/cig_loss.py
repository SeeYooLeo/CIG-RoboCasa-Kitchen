import torch
import torch.nn.functional as F


def compute_fm_energy(model, observation, actions, noise, time) -> torch.Tensor:
    losses = model(observation, actions, noise=noise, time=time)
    return losses.mean(dim=(1, 2))


def compute_fm_energy_cached(
    model,
    img_cache,
    lang_tokens,
    lang_masks,
    state,
    actions,
    noise,
    time,
    detach_prefix=False,
) -> torch.Tensor:
    losses = model.forward_with_cached_images(
        img_cache,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise,
        time,
        detach_prefix=detach_prefix,
    )
    return losses.mean(dim=(1, 2))


def compute_cig_loss(pos_energy, neg_energies, tau_c=1.0) -> torch.Tensor:
    scores = [-pos_energy]
    for neg_e in neg_energies:
        scores.append(-neg_e)
    logits = torch.stack(scores, dim=1).float() / tau_c
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
