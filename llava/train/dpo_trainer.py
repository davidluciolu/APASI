from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from torch.nn import Module
from llava.train.dpo_data_utils import get_batch_logps

from llava.train.llava_trainer import LLaVATrainer
from llava.constants import IGNORE_INDEX


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )

    # pad labels for image seq, labels.shape == output.logits.shape
    # model_name = model.module.__class__.__name__    # model is a DeepSpeedEngine
    # if model_name == 'PeftModelForCausalLM':
    #     _, _, _, _, _, labels = model.module.base_model.model.prepare_inputs_labels_for_multimodal(
    #         input_ids=input_ids,
    #         position_ids=None,
    #         attention_mask=attention_mask,
    #         past_key_values=None,
    #         labels=labels,
    #         images=images
    #     )
    # else:
    #     _, _, _, _, _, labels = model.module.prepare_inputs_labels_for_multimodal(
    #         input_ids=input_ids,
    #         position_ids=None,
    #         attention_mask=attention_mask,
    #         past_key_values=None,
    #         labels=labels,
    #         images=images
    #     )
    pad_len = output.logits.shape[1] - labels.shape[1]
    labels = torch.nn.functional.pad(labels, pad=(pad_len, 0), value=IGNORE_INDEX)

    if token_weighted:
        token_log_prob = get_batch_logps(output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = get_batch_logps(output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob

def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != IGNORE_INDEX)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp

class DPOTrainer(LLaVATrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):

        data_dict = inputs
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')

        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')

        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')

        ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
        ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
        ref_win_logp = data_dict.pop('ref_win_logp')
        ref_rej_logp = data_dict.pop('ref_rej_logp')
        ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
        ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
        if self.args.dpo_use_average:
            ref_win_logp = ref_win_avg_logp
            ref_rej_logp = ref_rej_avg_logp

        beta = data_dict.pop('beta')
        images = data_dict.pop('images')

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        concatenated_images = torch.cat([images, images], dim=0)

        win_token_weight = data_dict.pop('win_token_weight')
        rej_token_weight = data_dict.pop('rej_token_weight')
        concatenated_token_weight = data_dict.pop('concatenated_token_weight')

        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=self.args.dpo_token_weighted,
                                        dpo_use_average=self.args.dpo_use_average,
                                        **data_dict)
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        if self.args.dpo_token_weighted:
            ref_win_logp = compute_weighted_logp(ref_win_per_token_logp, win_labels, win_token_weight, self.args.dpo_use_average)
            ref_rej_logp = compute_weighted_logp(ref_rej_per_token_logp, rej_labels, rej_token_weight, self.args.dpo_use_average)
            concatenated_logp = compute_weighted_logp(concatenated_logp, concatenated_labels,concatenated_token_weight, self.args.dpo_use_average)

            if torch.any(torch.isnan(ref_win_logp)):
                print(f'ref_win_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(ref_rej_logp)):
                print(f'ref_rej_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(concatenated_logp)):
                print(f'concatenated_logp fail', flush=True)
                exit()


        policy_win_logp, policy_rej_logp = concatenated_logp.split([win_size, rej_size])


        if self.args.past_index >= 0:
            raise NotImplementedError

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # loss = losses.mean()

        # do SFT
        # loss = - policy_win_logp.mean()
        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()
        # loss = DPO_weight * losses.mean() - SFT_weight * policy_rej_logp.mean()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(policy_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(policy_win_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_rejected'] = self._nested_gather(ref_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_chosen'] = self._nested_gather(ref_win_logp.mean()).mean().item()
        # metrics[f'batch_size'] = len(win_labels)
        self.log(metrics)

        return loss
