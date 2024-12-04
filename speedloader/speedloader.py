from abc import ABC

import torch


class StagedMixin(ABC):
    def init_mixin(cls, **kwargs):
        cls.hidden_stream = torch.cuda.Stream()
        cls.hook_stream = torch.cuda.Stream()
        cls.past_key_values_length = 0
        cls.cache_swapper = None
        cls.hidden_swapper = None
        cls.batch_size = kwargs.get("batch_size", 64)
        cls.sub_batch_size = kwargs.get("sub_batch_size", 2)
        cls.num_sub_batches = cls.batch_size // cls.sub_batch_size
        if hasattr(kwargs, "num_sub_batches"):
            assert cls.num_sub_batches == kwargs["num_sub_batches"], "Number of sub-batches must be consistent."
        cls.n_live_blocks = 1
        cls.n_layers = len(cls.layers)

    def print_meta(self):
        print(f"Batch size: {self.batch_size}")
        print(f"Sub-batch size: {self.sub_batch_size}")
        print(f"Number of sub-batches: {self.num_sub_batches}")
        print(f"Number of live blocks: {self.n_live_blocks}")
        print(f"Number of layers: {self.n_layers}")

    def split_batch(self, *args, **kwargs):
        # Helper function to split a tensor into sub-batches
        def split_tensor(tensor, sub_batch_size):
            tensor_size = tensor.shape[0]
            sub_batches = [tensor[i : i + sub_batch_size] for i in range(0, tensor_size, sub_batch_size)]
            return sub_batches

        # Helper function to handle non-batched inputs
        def repeat_input(input, num_sub_batches):
            return [input] * num_sub_batches

        # Determine the number of sub-batches
        num_sub_batches = None
        for arg in args:
            if isinstance(arg, torch.Tensor) or isinstance(arg, list):
                if num_sub_batches is None:
                    batch_size = arg.shape[0]
                    # assert (batch_size % self.batch_size) == 0, f"Batch size {batch_size} must be divisible by sub_batch_size {self.sub_batch_size}. Result: {batch_size % self.sub_batch_size}"
                    num_sub_batches = arg.shape[0] // self.sub_batch_size
                    break

        for value in kwargs.values():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                if num_sub_batches is None:
                    batch_size = value.shape[0]
                    assert batch_size % self.batch_size == 0, "Batch size must be divisible by sub_batch_size."
                    num_sub_batches = value.shape[0] // self.sub_batch_size
                    break
        self.num_sub_batches = num_sub_batches
        self.batch_size = self.num_sub_batches * self.sub_batch_size
        if num_sub_batches is None:
            raise ValueError("No batched inputs found.")

        # Process args
        split_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) or isinstance(arg, list):
                split_args.append(split_tensor(arg, self.sub_batch_size))
            else:
                split_args.append(repeat_input(arg, num_sub_batches))

        # Process kwargs
        split_kwargs = {key: [] for key in kwargs}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                split_kwargs[key] = split_tensor(value, self.sub_batch_size)
            else:
                split_kwargs[key] = repeat_input(value, num_sub_batches)

        # Combine split_args and split_kwargs into sub-batches
        sub_batches = []
        for i in range(num_sub_batches):
            sub_batch_args = [arg[i] for arg in split_args]
            sub_batch_kwargs = {key: value[i] for key, value in split_kwargs.items()}
            sub_batches.append((sub_batch_args, sub_batch_kwargs))
        return sub_batches

    def embed_prologue(self, batch):
        if self.training:
            hidden_state = batch.pop("hidden_state", None)
            self.hidden_swapper.forward_prologue(hidden_state)
        return batch

    def embed_epilogue(self, batch, **kwargs):
        for k, v in kwargs.items():
            batch[k] = v
        if self.training:
            hidden_state = batch.pop("hidden_out")
            hidden_state_prev = batch.pop("hidden_state", None)
            hidden_state.is_hidden = True
            hidden_state.idx = kwargs["idx"]
            batch["hidden_state"] = self.hidden_swapper.forward_epilogue(hidden_state, hidden_state_prev)
        else:
            batch["hidden_state"] = batch.pop("hidden_out")
        return batch

    def block_prologue(self, batch):
        ## Execute prologue for block
        ## Before:
        ## Setup bid and lid for batch
        ## After:
        ## Assign return value with locals()[k] = v
        bid = batch["bid"]
        lid = batch["lid"]
        hidden_state = batch.pop("hidden_state", None)
        if not self.training:
            self.cache_swapper.step_prologue(bid, lid, self.n_live_blocks, self.past_key_values_length)
            k = self.cache_swapper.current_cache[:, 0]
            v = self.cache_swapper.current_cache[:, 1]
            batch["past_key_values"].key_cache[lid : lid + self.n_live_blocks] = list(k)
            batch["past_key_values"].value_cache[lid : lid + self.n_live_blocks] = list(v)
        else:
            hidden_state = self.hidden_swapper.forward_prologue(hidden_state)
            hidden_state.is_hidden = True
            batch["past_key_values"] = None
            batch["hidden_state"] = hidden_state
        return batch

    def block_epilogue(self, past_key_values, batch, **kwargs):
        ## After:
        ## Assign return value with locals()[k] = v
        ## Set batch['hidden_state'] = None if not using cache
        for k, v in kwargs.items():
            batch[k] = v
        lid = batch["lid"]
        if self.config.use_cache:
            k = torch.cat([past_key_values[lid + i][0] for i in range(self.n_live_blocks)], dim=0)
            v = torch.cat([past_key_values[lid + i][1] for i in range(self.n_live_blocks)], dim=0)
            kv = torch.cat([k, v], dim=0).permute(1, 0, 2, 3, 4, 5)
            self.cache_swapper.prev_cache = kv
            self.past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        else:
            hidden_state_prev = batch.pop("hidden_state", None)
            hidden_state = batch.pop("hidden_out")
            hidden_state = self.hidden_swapper.forward_epilogue(hidden_state, hidden_state_prev)
            batch["hidden_state"] = hidden_state
        self.hook_stream.synchronize()
        return batch

    def output_prologue(self, hidden_state):
        if not self.config.use_cache:
            hidden_state = self.hidden_swapper.forward_prologue(hidden_state)
        return hidden_state

    def output_epilogue(self, batch, **kwargs):
        for k, v in kwargs.items():
            batch[k] = v
        loss = batch.get("loss", None)
        hidden_state = batch.pop("hidden_state")
        if loss is not None and hidden_state.requires_grad and not self.config.use_cache:
            loss.is_loss = True
            hidden_state = self.hidden_swapper.forward_epilogue(loss)
            batch["hidden_state"] = hidden_state
        if not self.output_logits:
            batch["logits"] = None
        self.hook_stream.synchronize()
        return batch

    def reduce_outputs(self, batches):
        loss = None
        logits = None
        if batches[-1].get("loss", None) is not None:
            loss = batches[-1]["loss"]
        if batches[-1].get("logits", None) is not None:
            logits = torch.cat([batch["logits"] for batch in batches], dim=0)
        return (loss, logits)
