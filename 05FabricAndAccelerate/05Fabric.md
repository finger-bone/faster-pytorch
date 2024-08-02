# Fabric and Accelerate for Existing PyTorch Code

## Introduction

Previous parts discussed two complex acceleration framework, deepspeed and fair scale. To use them with ease, you had better use them with PyTorch Lightning. However, if you want to use PyTorch directly, you can use Fabric. It stays in the middle between PyTorch Lightning with complete management of the process, and PyTorch with manual management. Using Fabric, you can convert your existing PyTorch code to a distributed version with minimal changes.

In addition, accelerate from hugging face has a similar role to Fabric. Since both of them are simple to use, we will introduce them together.

## Converting from PyTorch to Fabric

It is extremely simple.

First setup fabric, the parameters are the same as PyTorch Lightning.

```python
from lightning.fabric import Fabric

fabric = Fabric(accelerator="cuda", devices=8, strategy="ddp")
```

If you need to run distributed training, you should add

```python
fabric.launch()
```

If not, use ignore the above line.

Then, setup the model and dataset.

```python
model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)
```

Remove all the manual movement of any tensor because Fabric will handle it.

Lastly, use the backward of Fabric.

```python
fabric.backward(loss)
```

That's it. You can use Fabric with minimal changes to your existing PyTorch code.

## Converting from PyTorch to Accelerate

Using hugging face Accelerate is even easier.

```python
from accelerate import Accelerator
accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
```

Also remove all `to` calls.

Later, use the loss of Accelerate.

```python
accelerator.backward(loss)
```

Accelerate also supports distributed training.

First config with accelerate cli.

```bash
accelerate config
```

Then, run the training script with `accelerate`.

```bash
accelerate launch script.py
```

As for the configuration, it supports many other options, just check the documentation.

## Conclusion

This part is rather short because Fabric and Accelerate are simple to use. They are designed to be used with minimal changes to existing PyTorch code. If you are not using PyTorch Lightning, you can use Fabric or Accelerate to accelerate your training process.

This series will temporarily end here. As a quick guide, this series introduced the most popular acceleration frameworks for PyTorch. If you are using PyTorch, you can use one of them to accelerate your training process.

As for temporarily, because torch-ignite is something interesting yet not stabled enough to be introduced. It is a high-level library for PyTorch, similar to PyTorch Lightning. It is worth checking out if you are interested in PyTorch acceleration. When it reaches a stable version, we will introduce it in this series.
