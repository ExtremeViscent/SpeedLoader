import inspect
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from speedloader.utils import async_save_on_cpu, split_batch

from .speedloader import StagedMixin

logger = logging.getLogger(__name__)


class LlamaSL(StagedMixin, LlamaModel):
    def __init__(self, config, **kwargs):
        LlamaModel.__init__(self, config)
        StagedMixin.__init__(self)
        self.init_mixin(**kwargs)

    def batch_collate(self, *args, **kwargs):
        signature = inspect.signature(LlamaModel.forward)
        bound_args = signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def forward_decoder_layers(
        self,
        hidden_states,
        batch,
    ):
        lid = batch.pop("lid", 0)
        for i in range(lid, min(lid + self.n_live_blocks, self.n_layers)):
            decoder_layer = self.layers[i]
            hidden_states = decoder_layer(hidden_states, **batch)
            if i == self.n_layers - 1:
                hidden_states = self.norm(hidden_states)
        batch["lid"] = lid
        return hidden_states, batch

    def forward_embed(
        self,
        batch: dict,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Any preprocessing and embedding of the input_ids should be done here.

        batch (dict): The batch object reference.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        if use_cache and not isinstance(past_key_values, Cache):
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # pack states in batch
        batch["attention_mask"] = causal_mask
        batch["position_embeddings"] = position_embeddings
        batch["past_key_values"] = past_key_values
        batch["inputs_embeds"] = inputs_embeds
        batch["cache_position"] = cache_position
        batch["hidden_out"] = hidden_states
        batch["position_ids"] = position_ids

        return batch

    def forward_decoder_layers(
        self,
        hidden_states,
        batch,
    ):
        lid = batch["lid"]
        layers = self.layers[lid : min(lid + self.n_live_blocks, self.n_layers)]
        signature = inspect.signature(LlamaModel.forward)
        valid_args = signature.parameters.keys()
        bound_args = {k: v for k, v in batch.items() if k in valid_args}
        for i, decoder_layer in enumerate(layers):
            bound_args["hidden_states"] = decoder_layer(hidden_states, **bound_args)[0]
        batch["hidden_out"] = bound_args["hidden_states"]
        return batch

    def forward(
        self,
        *args,
        **kwargs,
    ):
        batch = self.batch_collate(*args, **kwargs)
        batches = split_batch(**batch, sub_batch_size=self.sub_batch_size, batch_size=self.batch_size)
        batches = [self.batch_collate(*b[0], **b[1]) for b in batches]

        # Embedding
        for i in range(self.num_sub_batches):
            batches[i] = self.embed_prologue(batches[i])
            batches[i] = self.forward_embed(
                batches[i],
                **{k: v for k, v in batches[i].items() if k in inspect.signature(self.forward_embed).parameters},
            )
            batches[i] = self.embed_epilogue(batches[i], idx=i)

        # Decoder layers
        with async_save_on_cpu(self.hidden_swapper):
            for lid in range(0, self.n_layers, self.n_live_blocks):
                for i in range(self.num_sub_batches):
                    batches[i]["lid"] = lid
                    batches[i]["bid"] = i
                    batches[i] = self.block_prologue(batches[i])
                    hidden_state = batches[i].pop("hidden_state")
                    batches[i] = torch.utils.checkpoint.checkpoint(
                        self.forward_decoder_layers,
                        hidden_state,
                        batches[i],
                        use_reentrant=False,
                    )
                    batches[i]["hidden_state"] = hidden_state
                    del hidden_state
                    batches[i] = self.block_epilogue(None, batches[i])

        # Output
        return batches


class LlamaForCausalLMSL(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaSL(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def head(self, hidden_states, labels, return_logits=False):
        torch.cuda.empty_cache()
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss, logits if return_logits else None

    def forward(
        self,
        *args,
        **kwargs,
    ):
        labels = kwargs.pop("labels", None)
        labels = split_batch(labels, sub_batch_size=self.model.sub_batch_size, batch_size=self.model.batch_size)
        if self.training:
            self.model.hidden_swapper.resize_buffer(
                labels[0][0][0].shape[-1],
                sub_batch_size=self.model.sub_batch_size,
                effective_batch_size=self.model.batch_size,
            )
        batches = self.model(*args, **kwargs)

        # Head
        hidden_states = batches[-1]["hidden_state"]
        for i in range(self.model.num_sub_batches):
            label = labels[i][0][0] if labels is not None else None
            hidden_states = self.model.output_prologue(hidden_states)
            loss, logits = torch.utils.checkpoint.checkpoint(
                self.head,
                hidden_states,
                label,
                use_reentrant=False,
            )
            batches[i] = self.model.output_epilogue(batches[i], loss=loss, logits=logits)
            hidden_states = batches[i].pop("hidden_state")
        loss, logits = self.model.reduce_outputs(batches)

        del batches
        if self.training:
            self.model.hidden_swapper.pre_backward()

        if not kwargs.get("return_dict", False):
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
