import torch
import torch.nn as nn
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.nn import CrossEntropyLoss

class UnifiedQG(nn.Module):
    def __init__(self, config):
        super(UnifiedQG, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name_or_path)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def get_loss_for_examplars(self, input_ids, input_attn_mask, target_ids, target_attn_mask):
        """专门用来计算loss for examplar选择的，此处手动取平均，而不是默认的求和"""
        lm_labels = target_ids
       # decoder_input_ids = shift_tokens_right(target_ids, self.tokenizer.pad_token_id, 0)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100  # -100就不会被计算loss
        # 进行一遍forward函数,继承自nn.Module
        outputs = self(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
           # decoder_input_ids=decoder_input_ids,
            labels=lm_labels,
            decoder_attention_mask=target_attn_mask
        )

        logits = outputs[1]  # (batch_size, sequence_length, config.vocab_size)
        ce = CrossEntropyLoss(reduction='none')
        logits = logits.permute(0,2,1)
        loss = ce(logits, lm_labels)  # [B, seq_len]
        length = (lm_labels != -100).float().sum(-1)  # [B]
        loss = loss.sum(dim=-1) / length  # [B]

        return loss

    def get_loss(self, input_ids, input_attn_mask, target_ids, target_attn_mask):

        lm_labels = target_ids
       # decoder_input_ids = shift_tokens_right(target_ids, self.tokenizer.pad_token_id, 0)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100  # -100就不会被计算loss
        outputs = self(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
           # decoder_input_ids=decoder_input_ids,
            labels=lm_labels,
            decoder_attention_mask=target_attn_mask
        )
        loss = outputs[0]
        return loss

    def ewc_loss(self, loss, ewc_loss, importance):
        """Add ewc regularization loss to original loss"""
        loss = loss + importance * ewc_loss

        return loss


