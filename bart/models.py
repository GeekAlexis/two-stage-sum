import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration


class BartParaphraser(nn.Module):
    def __init__(self, arch='facebook/bart-large', attention_dropout=0.1):
        super().__init__()
        self.arch = arch
        self.bart = BartForConditionalGeneration.from_pretrained(self.arch,
                                                                 attention_dropout=attention_dropout)

    def forward(self, input_ids=None, **kwargs):
        return self.bart(input_ids=input_ids, **kwargs)

    def generate(self, input_ids=None, **kwargs):
        return self.bart.generate(input_ids=input_ids, **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels):
        # DataCollatorForSeq2Seq needs this to convert labels to decoder_input_ids
        return self.bart.prepare_decoder_input_ids_from_labels(labels)


class LabelSmoothingLoss(nn.Module):
    """
    Adds label-smoothing to cross entropy loss.
    https://github.com/huggingface/transformers/blob/b66c5ab20c8bb08d52cb840382498f936ea8da03/src/transformers/trainer_pt_utils.py#L447-L483
    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    def __init__(self, epsilon=0.0, ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        log_probs = -F.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss