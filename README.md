# SpeedLoader
With the surging growth of model parameters, foundation models pose unprecedented challenges to traditional computational infrastructures. These large models inherently require substantial accelerator memory to accommodate massive tensors during pre-training, fine-tuning, and even inference stages, making it even more challenging to deploy a model with restricted computational resources. Given this challenge, distribution and offloading the model states are two major solutions. Partitioning the required states to participating workers, and storing them in lower speed media, such as host DRAM and block devices, largely alleviate the accelerator memory pressure. However, the prohibitive costs of tensor communication render it a theoretically plausible yet practically inefficient solution. Previous efforts to improve efficiency include maximizing rematerialization and employing chunk-based tensor management to reduce host-device communication. Despite these efforts, the reported training throughput only achieves 36.54% of model FLOPs utilization (MFUs), still not comparable to full on-device training. In this work, we redesign the data flow of heterogeneous hardware and sharded model training to minimize the excessive communication overhead. Our proposed scheme significantly enhances training and inference throughput of large language models under restrictive computational resources. We confirmed a large leap in effective compute time by looking into the kernel-level runtime behavior of our trials, where the MFUs can achieve up to 51%. Compared to the state-of-the-art approach, our framework robustly achieves remarkable speedups from 3x to 30x in multiple distributed heterogeneous training setups and inference speedups of 1.5x to 2.35x without compromising arithmetic precision.

## Usage

### Prepare the environment

Install the SpeedLoader package by running the following command.

```bash
pip install .
```

### Run the minimal example

To demonstrate the usage of SpeedLoader, we provide a minimal example based on a 1.3B LLaMA model. The following command will run the minimal example with 1 GPU.

```bash
deepspeed \
--num_gpus 1 example.py \
--sub-batch-size 8 \
--num-sub-batches 8 \
--num-live-blocks 1 \
--sequence-length 96 \
--gradient-accumulation-steps 1
```

## Adaptation to other models

We have provided an example of LLaMA adapation. It is easy to adapt SpeedLoader to other Transformer-based models.

Essentially, you need to complete the following steps:

1. Define a new model class that inherits from `StagedMixin` and original model class.

    The `StagedMixin.init_mixin()` method should be called in the `__init__` method of the new model class.

2. Modify the forward method of the new model class.

    Copy the forward method of the original model class make adjustments accordinglly as shown in the example. Modified parts are marked with comments.

3. For causal models, a modified version of causal model is also required in addition to the base model.
