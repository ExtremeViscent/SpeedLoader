import logging
import os

import numpy as np
import psutil
import torch
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor


def setup_logging():
    log_level_str = os.environ.get("BB_LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)
    logger = logging.getLogger("balance_beam")
    logger.setLevel(log_level)
    return logger


logger = setup_logging()


def split_batch(*args, **kwargs):
    # Sanity checks
    batch_size = kwargs.pop("batch_size", None)
    sub_batch_size = kwargs.pop("sub_batch_size", None)
    assert batch_size is not None, "batch_size must be provided."
    assert sub_batch_size is not None, "sub_batch_size must be provided."

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
                num_sub_batches = arg.shape[0] // sub_batch_size
                break
    if num_sub_batches is None:
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                if num_sub_batches is None:
                    batch_size = value.shape[0]
                    assert batch_size % batch_size == 0, "Batch size must be divisible by sub_batch_size."
                    num_sub_batches = value.shape[0] // sub_batch_size
                    break
    batch_size = num_sub_batches * sub_batch_size
    if num_sub_batches is None:
        raise ValueError("No batched inputs found.")

    # Process args
    split_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) or isinstance(arg, list):
            split_args.append(split_tensor(arg, sub_batch_size))
        else:
            split_args.append(repeat_input(arg, num_sub_batches))

    # Process kwargs
    split_kwargs = {key: [] for key in kwargs}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) or isinstance(value, list):
            split_kwargs[key] = split_tensor(value, sub_batch_size)
        else:
            split_kwargs[key] = repeat_input(value, num_sub_batches)

    # Combine split_args and split_kwargs into sub-batches
    sub_batches = []
    for i in range(num_sub_batches):
        sub_batch_args = [arg[i] for arg in split_args]
        sub_batch_kwargs = {key: value[i] for key, value in split_kwargs.items()}
        sub_batches.append((sub_batch_args, sub_batch_kwargs))
    return sub_batches


def init_hidden_buffer_pool(hidden_size, max_seq_len, num_ckpts, effective_batch_size, sub_batch_size):
    buffer_size = hidden_size * max_seq_len * num_ckpts * effective_batch_size
    buffer_pool = torch.empty(buffer_size, dtype=torch.half, device=torch.device("cpu"), pin_memory=True)
    return buffer_pool


def get_hidden_buffer(buffer_pool, hidden_size, seq_len, num_ckpts, effective_batch_size, sub_batch_size):
    num_sub_batches = effective_batch_size // sub_batch_size
    buffer_size = hidden_size * seq_len * num_ckpts * num_sub_batches * sub_batch_size
    buffer = buffer_pool.narrow(0, 0, buffer_size)
    buffer = buffer.view(num_sub_batches, num_ckpts, sub_batch_size, seq_len, hidden_size)
    return buffer


class async_save_on_cpu(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, hidden_swapper, pin_memory=True):
        self.hidden_swapper = hidden_swapper

        def pack_to_cpu(tensor):
            if getattr(tensor, "is_hidden", False):
                idx = tensor.idx
                packed = torch.empty(
                    1,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    pin_memory=(torch.cuda.is_available() and not tensor.is_sparse),
                )
                packed.is_hidden = True
                packed.idx = idx
                return (tensor.device, packed)
            else:
                if not pin_memory:
                    return (tensor.device, tensor.cpu())
                packed = torch.empty(
                    tensor.size(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    pin_memory=(torch.cuda.is_available() and not tensor.is_sparse),
                )
                packed.copy_(tensor)
                return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed
            if getattr(tensor, "is_hidden", False):
                unpacked = self.hidden_swapper.backward_prologue_hidden()
                return unpacked
            else:
                return tensor.to(device, non_blocking=True)

        super().__init__(pack_to_cpu, unpack_from_cpu)


class AsyncHiddenSwapper:
    def __init__(
        self,
        hidden_size,
        max_sequence_length,
        effective_batch_size,
        sub_batch_size,
        num_ckpts,
        training=False,
        dtype=torch.half,
        quantization=False,
    ):
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.effective_batch_size = effective_batch_size
        self.sub_batch_size = sub_batch_size
        self.num_ckpts = num_ckpts if training else 1
        self.stream = torch.cuda.Stream()
        self.num_sub_batches = self.effective_batch_size // self.sub_batch_size
        self.buffer_size = self.hidden_size * self.max_sequence_length
        self.buffer_size *= self.effective_batch_size * self.num_ckpts
        self.buffer_size = int(self.buffer_size)
        self.training = training
        self.dtype = dtype
        self.quantization = quantization
        self.quantizers = [None for _ in range(self.num_ckpts * self.num_sub_batches)]
        print(
            f"Initializing {dtype} buffer pool with size: {self.buffer_size * torch.finfo(dtype).bits//8 / 1024**3} GB"
        )
        # Show resident memory before allocating buffer pool
        import psutil

        psutil.Process(os.getpid())
        # print(f'Before buffer pool allocation: {process.memory_info().rss / 1024**3} GB')
        for i in range(self.num_ckpts * self.num_sub_batches):
            if not self.quantization:
                break
            quant_desc = QuantDescriptor(num_bits=8, fake_quant=False, axis=-1)
            self.quantizers[i] = TensorQuantizer(quant_desc)
        # print(f'After quantizer allocation: {process.memory_info().rss / 1024**3} GB')
        self.buffer_pool = torch.empty(
            self.buffer_size,
            dtype=self.dtype if not self.quantization else torch.int8,
            device="cpu",
            requires_grad=False,
            pin_memory=True,
        )
        # Show resident memory after allocating buffer pool
        # print(f'After buffer pool allocation: {process.memory_info().rss / 1024**3} GB')
        self.buffer = self.buffer_pool.view(
            self.num_ckpts * self.num_sub_batches, self.sub_batch_size, self.max_sequence_length, self.hidden_size
        )
        # print(f'Afer buffer pool view: {process.memory_info().rss / 1024**3} GB')
        self.dummies = [None for _ in range(self.num_ckpts * self.num_sub_batches)]
        self.prev_hidden = None
        self.current_hidden = None
        self.current_hidden_ = None
        self.next_hidden = None
        self.next_hidden_ = None
        self.prev_grad = None
        self.current_grad = None
        self.next_grad = None
        self.current_idx = 0
        self.prev_idx = self.current_idx + self.num_sub_batches - 1
        self.next_idx = self.current_idx + 1
        self.current_grad_idx = -1
        self.prev_grad_idx = self.current_grad_idx - self.num_sub_batches + 1
        self.next_grad_idx = self.current_grad_idx - 1
        self.hidden_initialized = torch.zeros(
            self.num_ckpts * self.num_sub_batches,
            dtype=torch.bool,
            device=torch.device("cpu"),
            requires_grad=False,
        )
        self.grad_initialized = torch.zeros(
            self.num_ckpts * self.num_sub_batches,
            dtype=torch.bool,
            device=torch.device("cpu"),
            requires_grad=False,
        )
        self.grad_stack = []

    def reset(self):
        self.prev_hidden = None
        self.current_hidden = None
        self.current_hidden_ = None
        self.next_hidden = None
        self.next_hidden_ = None
        self.prev_grad = None
        self.current_grad = None
        self.next_grad = None
        self.current_idx = 0
        self.prev_idx = self.current_idx + self.num_sub_batches - 1
        self.next_idx = self.current_idx + 1
        self.current_grad_idx = -1
        self.prev_grad_idx = self.current_grad_idx - self.num_sub_batches + 1
        self.next_grad_idx = self.current_grad_idx - 1
        self.hidden_initialized = torch.zeros(
            self.num_ckpts * self.num_sub_batches,
            dtype=torch.bool,
            device=torch.device("cpu"),
            requires_grad=False,
        )

        for i in range(self.num_ckpts * self.num_sub_batches):
            if not self.quantization:
                break
            if i < len(self.quantizers):
                self.quantizers[i] = TensorQuantizer(QuantDescriptor(num_bits=8, fake_quant=False, axis=-1))
            else:
                self.quantizers.append(TensorQuantizer(QuantDescriptor(num_bits=8, fake_quant=False, axis=-1)))

    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            with torch.no_grad():
                return tensor_quant.tensor_quant(x.detach(), x.abs().max(), 8)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    class Dequantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale):
            with torch.no_grad():
                return x.detach() / scale

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def quantize(self, x, idx):
        if x is None:
            return x
        x_q = self.quantizers[idx](x).to(torch.int8)
        # print(f'quantize: {idx}, value: {x_q[0,0,0]}')
        return x_q

    def dequantize(self, x, idx):
        if x is None:
            return x
        x = x / self.quantizers[idx].scale
        # print(f'dequantize: {idx}, dtype: {x.dtype}, value: {x[0,0,0]}')
        return x

    def resize_buffer(self, seq_len=None, effective_batch_size=None, sub_batch_size=None, num_ckpts=None):
        self.seq_len = seq_len or self.seq_len
        self.effective_batch_size = effective_batch_size or self.effective_batch_size
        self.sub_batch_size = sub_batch_size or self.sub_batch_size
        self.num_sub_batches = self.effective_batch_size // self.sub_batch_size
        self.num_ckpts = num_ckpts or self.num_ckpts
        new_buffer_size = self.hidden_size * self.seq_len * self.effective_batch_size * self.num_ckpts
        new_buffer_size = int(new_buffer_size)
        if self.buffer_size < new_buffer_size:
            self.buffer_size = new_buffer_size
            self.buffer_pool.resize_(new_buffer_size)
        self.buffer = self.buffer_pool.narrow(0, 0, new_buffer_size).view(
            self.num_ckpts * self.num_sub_batches, self.sub_batch_size, self.seq_len, self.hidden_size
        )
        self.reset()

    def get_buffer_idx(self, cid, bid):
        return cid * self.num_sub_batches + bid

    def indices_increment(self):
        self.current_idx += 1
        self.prev_idx += 1
        self.next_idx += 1
        if self.current_idx == self.num_ckpts * self.num_sub_batches:
            self.current_idx = 0
        if self.prev_idx == self.num_ckpts * self.num_sub_batches:
            self.prev_idx = 0
        if self.next_idx == self.num_ckpts * self.num_sub_batches:
            self.next_idx = 0

    def get_cid_bid(self, idx):
        return idx // self.num_sub_batches, idx % self.num_sub_batches

    def offload_hidden(self, idx):
        # Abort offloading if conflict or empty or out of range
        if (
            self.next_idx == idx
            or self.current_idx == idx
            or idx < 0
            or idx >= self.num_ckpts * self.num_sub_batches
            or self.prev_hidden is None
        ):
            # self.prev_idx = -1
            self.prev_hidden = None
            return
        if getattr(self.prev_hidden, "is_loss", False) or not getattr(self.prev_hidden, "is_hidden", False):
            return
        # print(f'offload_hidden: {idx}')
        # Get cid and bid
        cid, bid = self.get_cid_bid(idx)
        # Get buffer
        buffer = self.buffer[idx]
        # Copy hidden to buffer
        with torch.cuda.stream(self.stream):
            if self.quantization:
                buffer.copy_(self.quantizers[idx](self.prev_hidden).to(torch.int8), non_blocking=True)
            else:
                buffer.copy_(self.prev_hidden, non_blocking=True)
        # Mark buffer as initialized
        self.hidden_initialized[idx] = True
        # Set hidden attributes
        buffer.bid = bid
        buffer.cid = cid

    def fetch_hidden(self, idx):
        # Abort fetching if conflict or empty or out of range
        if (
            self.prev_idx == idx
            or self.current_idx == idx
            or idx < 0
            or idx >= (self.num_ckpts - 1) * self.num_sub_batches
            or self.hidden_initialized[idx] == False
        ):
            return None
        # Get buffer
        # print(f'fetch_hidden: {idx}')
        buffer = self.buffer[idx]
        with torch.cuda.stream(self.stream):
            # Copy buffer to hidden
            if self.quantization:
                ret = buffer.to(torch.device("cuda"), non_blocking=True) / self.quantizers[idx].scale
            else:
                ret = buffer.to(torch.device("cuda"), non_blocking=True)
        # Set hidden attributes
        ret.idx = idx
        ret.is_hidden = True
        return ret

    def sync_fetch_hidden(self, idx):
        # Proceed only if not already fetched and already initialized
        if (self.current_hidden is None or self.current_idx != idx) and self.hidden_initialized[idx]:
            # Get buffer
            buffer = self.buffer[idx]
            # Copy buffer to hidden
            if self.quantization:
                self.current_hidden = buffer.to(torch.device("cuda"), non_blocking=False) / self.quantizers[idx].scale
            else:
                self.current_hidden = buffer.to(torch.device("cuda"), non_blocking=False)
            # Set hidden attributes
            self.current_hidden.idx = idx
            self.current_hidden.is_hidden = True
            # print(f'prefetch miss: {idx}')

    def synchronize(self):
        self.stream.synchronize()

    class AttachFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, src, dummy):
            return src

        @staticmethod
        def backward(ctx, grad_output):
            return None, grad_output

    class InputAttachFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, src, dst):
            ctx.null_dst = True if dst is None else False
            return src

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output if not ctx.null_dst else None

    class OutputAttachFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, src, loss):
            ctx.save_for_backward(loss)
            return src

        @staticmethod
        def backward(ctx, grad_output):
            (loss,) = ctx.saved_tensors
            return None, torch.tensor(1.0).to(grad_output.device)

    def forward_prologue(self, dst=None):
        # print(f'forward_prologue: {self.current_idx}, {self.next_idx} {self.prev_idx}')

        # Set current hidden
        self.current_hidden = self.next_hidden
        # Clear next hidden
        self.next_hidden = None

        # Offload previous hidden
        self.offload_hidden(self.prev_idx)
        # Fetch next hidden
        self.next_hidden = self.fetch_hidden(self.next_idx)
        # Verify current hidden
        self.sync_fetch_hidden(self.current_idx)

        if self.current_idx >= self.num_sub_batches:
            # Attach current hidden to dst
            if dst is not None:
                if self.current_idx > self.num_sub_batches * (self.num_ckpts - 2):
                    self.current_hidden = AsyncHiddenSwapper.OutputAttachFunction.apply(self.current_hidden, dst)
                else:
                    self.current_hidden = AsyncHiddenSwapper.AttachFunction.apply(self.current_hidden, dst)
                self.current_hidden.idx = self.current_idx
                self.current_hidden.is_hidden = True
                self.current_hidden = AsyncHiddenSwapper.ReplaceGrad(self).register(self.current_hidden)
        elif dst is not None:
            self.current_hidden = AsyncHiddenSwapper.InputAttachFunction.apply(self.current_hidden, dst)
            if self.current_hidden is not None:
                self.current_hidden.idx = self.current_idx
                self.current_hidden.is_hidden = True
                self.current_hidden = AsyncHiddenSwapper.ReplaceGrad(self).register(self.current_hidden)

        self.indices_increment()

        return self.current_hidden

    def forward_epilogue(self, prev_hidden, dst=None):
        torch.cuda.current_stream().synchronize()
        self.synchronize()
        if self.current_idx < self.num_sub_batches + 1:
            if self.current_idx < self.num_sub_batches:
                AsyncHiddenSwapper.ReplaceGrad(self).register(prev_hidden)
            prev_hidden = AsyncHiddenSwapper.InputAttachFunction.apply(prev_hidden, dst)
            prev_hidden.is_hidden = True
            prev_hidden.idx = self.prev_idx
        self.prev_hidden = prev_hidden.detach() if prev_hidden is not None else None
        self.prev_hidden.is_loss = getattr(prev_hidden, "is_loss", False)
        self.prev_hidden.is_hidden = (
            getattr(prev_hidden, "is_hidden", True) if not getattr(prev_hidden, "is_loss", False) else False
        )
        self.prev_hidden.idx = self.prev_idx
        # print(f'forward_epilogue: {self.current_idx}, {self.next_idx} {self.prev_idx}')
        return prev_hidden

    def pre_backward(self):
        torch.cuda.current_stream().synchronize()
        self.synchronize()
        self.current_idx = self.next_idx - 2
        self.next_idx = self.current_idx - 1
        self.current_grad_idx = self.current_idx + self.num_sub_batches - 1
        self.prev_grad_idx = self.current_grad_idx - self.num_sub_batches + 1
        self.next_grad_idx = self.current_grad_idx - 1
        self.current_hidden_ = self.current_hidden

    def offload_grad(self, idx):
        # Abort offloading if conflict or empty or out of range
        if (
            self.next_grad_idx == idx
            or self.current_grad_idx == idx
            or idx < 0
            or idx >= self.num_ckpts * self.num_sub_batches
            or self.prev_grad is None
        ):
            self.prev_grad = None
            return
        # get buffer
        buffer = self.buffer[idx]
        # Copy grad to buffer
        with torch.cuda.stream(self.stream):
            if self.quantization:
                self.quantizers[idx] = TensorQuantizer(QuantDescriptor(num_bits=8, fake_quant=False, axis=-1))
                buffer.copy_(self.quantizers[idx](self.prev_grad).to(torch.int8), non_blocking=True)
            else:
                buffer.copy_(self.prev_grad, non_blocking=True)
        # Mark buffer as initialized
        self.grad_initialized[idx] = True

    def fetch_grad(self, idx):
        # Abort fetching if conflict or empty or out of range
        if (
            self.prev_grad_idx == idx
            or self.current_grad_idx == idx
            or idx < 0
            or idx >= (self.num_ckpts - 1) * self.num_sub_batches
            or self.grad_initialized[idx] == False
        ):
            self.next_grad = None
            return
        # Get buffer
        # print(f'fetch_grad: {idx}')
        buffer = self.buffer[idx]
        # Copy buffer to grad
        with torch.cuda.stream(self.stream):
            if self.quantization:
                self.next_grad = buffer.to(torch.device("cuda"), non_blocking=True) / self.quantizers[idx].scale
            else:
                self.next_grad = buffer.to(torch.device("cuda"), non_blocking=True)
        # Set grad attributes
        self.next_grad.idx = idx

    def sync_fetch_grad(self, idx):
        # Proceed only if not already fetched and already initialized
        if (self.current_grad is None or self.current_grad_idx != idx) and self.grad_initialized[idx]:
            # Get buffer
            buffer = self.buffer[idx]
            # Copy buffer to grad
            if self.quantization:
                self.current_grad = buffer.to(torch.device("cuda"), non_blocking=False) / self.quantizers[idx].scale
            else:
                self.current_grad = buffer.to(torch.device("cuda"), non_blocking=False)
            # Set grad attributes
            self.current_grad.idx = idx
            # print(f'prefetch miss: {idx}')

    def indices_decrement(self):
        self.current_idx -= 1
        self.next_idx = self.current_idx - 1
        self.current_grad_idx -= 1
        self.prev_grad_idx = self.current_grad_idx - self.num_sub_batches + 1
        self.next_grad_idx = self.current_grad_idx - 1

    def indices_decrement_hidden(self):
        self.current_idx -= 1
        self.next_idx = self.current_idx - 1

    def indices_decrement_grad(self):
        self.current_grad_idx -= 1
        self.next_grad_idx = self.current_grad_idx - 1
        self.prev_grad_idx = self.current_grad_idx - self.num_sub_batches + 1

    def backward_prologue_hidden(self):
        # Set current hidden
        self.current_hidden = self.next_hidden
        # Clear next hidden
        self.next_hidden = None
        # Fetch next hidden
        self.next_hidden = self.fetch_hidden(self.next_idx)
        # Verify current hidden
        self.sync_fetch_hidden(self.current_idx)
        self.indices_decrement_hidden()
        return self.current_hidden

    def backward_prologue_grad(self, prev_grad):
        torch.cuda.current_stream().synchronize()
        self.synchronize()
        # Set current grad
        self.current_grad = self.next_grad
        # Clear next grad
        self.next_grad = None
        # Set prev grad
        self.prev_grad = prev_grad
        # Offload previous grad
        self.offload_grad(self.prev_grad_idx)
        # Fetch next grad
        self.fetch_grad(self.next_grad_idx)
        # Verify current grad
        self.sync_fetch_grad(self.current_grad_idx)
        self.indices_decrement_grad()
        return self.current_grad

    class ReplaceGrad:
        def __init__(self, hidden_swapper):
            self.hidden_swapper = hidden_swapper

            def hook(grad):
                ret = self.hidden_swapper.backward_prologue_grad(grad)
                return ret

            self.hook_fn = hook

        def register(self, x):
            if x.requires_grad:
                x.register_hook(self.hook_fn)
            return x


def get_max_batch_size(
    seq_len,
    model,
    training=False,
    num_ckpts=1,
):
    available_mem = psutil.virtual_memory().available
    hidden_size = model.config.hidden_size
    if not training:
        num_layers = model.config.num_hidden_layers
        per_sample_size = hidden_size * seq_len * num_layers * 2 * 2
    else:
        per_sample_size = hidden_size * seq_len * num_ckpts * 2
    max_batch_size = available_mem // per_sample_size * 0.8
    # Align batch size to multiple of 64
    max_batch_size = 64 * (max_batch_size // 64)
    return max_batch_size


def get_max_sub_batch_size(
    seq_len,
    model,
):
    current_device = torch.cuda.current_device()
    hidden_size = model.config.hidden_size
    live_param_size = hidden_size * hidden_size * 12 * 2
    available_mem = torch.cuda.get_device_properties(current_device).total_memory - live_param_size
    hidden_size = model.config.hidden_size
    per_sample_size = hidden_size * seq_len * 200 * 2
    max_sub_batch_size = available_mem // per_sample_size
    # Align batch size to power of 2
    max_sub_batch_size = 2 ** int(np.log2(max_sub_batch_size))
    return max_sub_batch_size
