# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic Composer on Databricks
# MAGIC This is based on the demo here: 
# MAGIC https://github.com/mosaicml/composer

# COMMAND ----------

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import mnist_model
from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast

# We need dist utilities for setting up distribution
from composer.utils import dist

# COMMAND ----------

# This is the raw code
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)

trainer = Trainer(
    model=mnist_model(num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",
    algorithms=[
        LabelSmoothing(smoothing=0.1),
        CutMix(alpha=1.0),
        ChannelsLast(),
        ]
)
trainer.fit()

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling on Databricks
# MAGIC Whilst Composer has it's own CLI it is just a wrapper around torchrun (much like hf accelerate too) 

# COMMAND ----------

# To make the code ready we wrap it in a function we can distribute

def main_train():

    # this is pretty much the same code just wrapped
    # We did add in the dist_sampler to make sure we shard the data correctly

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

    # Adding the distributed data sampler
    sampler = dist.get_sampler(dataset, shuffle=True)

    train_dataloader = DataLoader(dataset, batch_size=128, sampler=sampler)

    trainer = Trainer(
        model=mnist_model(num_classes=10),
        train_dataloader=train_dataloader,
        max_duration="2ep",
        algorithms=[
            LabelSmoothing(smoothing=0.1),
            CutMix(alpha=1.0),
            ChannelsLast(),
            ]
    )
    trainer.fit()

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling with Torchdistributor
# MAGIC We can scale the process on TorchDistributor 
    
# COMMAND ----------
    
from pyspark.ml.torch.distributor import TorchDistributor

distributor = TorchDistributor(num_processes=2, 
                                local_mode=True, 
                                use_gpu=True)
 
completed_trainer = distributor.run(main_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling with DeepspeedDistributor
# MAGIC Composer Supports Deepspeed as well 
    
# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

# the dict could also be a json file
deepspeed_dict = {
  "train_batch_size": 2048,
  "fp16": {"enabled": True},
  "zero)_optimization": {
    "stage": 1
  }
}

distributor = DeepspeedTorchDistributor(numGpus=2, nnodes=1, localMode=True, 
                                            useGpu=True, deepspeedConfig = deepspeed_dict)

completed_trainer = distributor.run(main_train)

# COMMAND ----------


