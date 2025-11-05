
# This tutorial guides you through the training of EuroBERT-210M model using the Optimus library, extending its language support by adding Finnish. We will cover installation, data preprocessing, and model training.
# 
# The tutorial is divided into two parts: training in pure Python or executing the Optimus library directly from the command line (useful for distributed settings and server training).
# 
# **Table of Contents**
# - 🐍 [Python](#python)
# - 💻 [Command Line](#command-line)
# 
# **Resources:**
# - 🤖 [EuroBERT](https://huggingface.co/EuroBERT)
# - 🚀 [Optimus Training Library](https://github.com/Nicolas-BZRD/EuroBERT)
# - 📄 [Paper](https://arxiv.org/pdf/2503.05500)
# - 📚 [Data Used](https://huggingface.co/datasets/Finnish-NLP/wikipedia_20230501_fi_cleaned)
# ## Preparing the Data
# For this tutorial, we’ll be using the Finnish Wikipedia dataset from [Hugging Face](https://huggingface.co/datasets/Finnish-NLP/wikipedia_20230501_fi_cleaned). The preprocessing steps will make sure the data is properly formatted for training.
# 
# **Steps:**
# 1. Download the dataset.
# 2. Tokenize the text with the EuroBERT tokenizer.
# 3. Pack the data **(optional)**.
# 4. Datamix.
# 
# To get started, we just need to import the `dataprocess` function like this:

import sys
import os
# Ensure we're using the local optimus module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimus import dataprocess

# ## 1. Downloading the Dataset
# 
# For efficient tokenization, Optimus uses raw data that can be easily downloaded from Hugging Face using `git clone`.

# ## 2. Tokenizing the Dataset

# The Finnish Wikipedia dataset consists of 11 columns, including "text", "id", and "url". These three columns match the expected format for our processing script for [Wikipedia dumps](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/optimus/dataprocess/dataset/wikipedia.py).
# 
# If you need to work with other datasets, check out the [existing dataset](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/optimus/dataprocess/dataset) scripts for examples on how to create a compatible processing script.

if __name__ == "__main__":

    import os
    import shutil

    # Clean up previous runs
    shutil.rmtree("tokenized", ignore_errors=True)
    shutil.rmtree("datamix", ignore_errors=True)
    shutil.rmtree("packed_greedy", ignore_errors=True)
    shutil.rmtree("packed_ffs", ignore_errors=True)
    shutil.rmtree("packed_ffd", ignore_errors=True)

    dataprocess.tokenize_dataset(
        input_dir="wikipedia_20230501_fi_cleaned",  # Path to the raw dataset
        tokenizer="EuroBERT/EuroBERT-210m",  # Path to the EuroBERT tokenizer model or HuggingFace model ID
        dataset="wikipedia",  # Dataset format (e.g., 'wikipedia')
        output_dir="tokenized",  # Directory where the tokenized data will be saved
        num_workers="max",  # Use the maximum available workers for parallel processing
        head=1,  # Sample only 1 record (~8134444 tokens)
        tiktoken=False  # Enable TikToken for efficient tokenization
    )

    import json
    with open("tokenized/metadata.json", "r") as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))

    # Additionally, we can take a look at the first sample of our tokenized data to make sure everything looks good.

    dataprocess.inspect_dataset(input_dir="tokenized", tokenizer="EuroBERT/EuroBERT-210m", num_samples=1)
    
    # ## 3. Pack our Data **(optional)**
    # We'll compare three packing algorithms:
    # 1. Greedy (fast, ~70-85% efficiency)
    # 2. First-Fit Shuffle (optimized, ~90-98% efficiency)
    # 3. First-Fit Decreasing (optimized, ~90-98% efficiency)
    
    print("\n" + "="*80)
    print("Testing GREEDY PACKING")
    print("="*80)
    
    dataprocess.pack_dataset(
        input_dir="tokenized",
        output_dir="packed_greedy",
        block_size=2048,
        packing_algorithm="greedy",  # Can also use "first_fit_decreasing"
        seed=42,
        num_workers=4
    )

    dataprocess.inspect_dataset(input_dir="packed_greedy/train", tokenizer="EuroBERT/EuroBERT-210m", num_samples=2, tiktoken=False)
    
    print("\nGreedy Packing Metadata:")
    with open("packed_greedy/packing_metadata.json", "r") as f:
        print(json.dumps(json.load(f), indent=2))
    
    print("\n" + "="*80)
    print("Testing OPTIMIZED PACKING (first_fit_shuffle algorithm)")
    print("="*80)
    
    dataprocess.pack_dataset(
        input_dir="tokenized",
        output_dir="packed_ffs",
        block_size=2048,
        packing_algorithm="first_fit_shuffle",  # Can also use "first_fit_decreasing"
        seed=42,
        num_workers=4
    )

    dataprocess.inspect_dataset(input_dir="packed_ffs/train", tokenizer="EuroBERT/EuroBERT-210m", num_samples=2, tiktoken=False)
    
    print("\nFirst-Fit Shuffle Metadata:")
    with open("packed_ffs/packing_metadata.json", "r") as f:
        print(json.dumps(json.load(f), indent=2))

    print("\n" + "="*80)
    print("Testing OPTIMIZED PACKING (first_fit_decreasing algorithm)")
    print("="*80)

    dataprocess.pack_dataset(
        input_dir="tokenized",
        output_dir="packed_ffd",
        block_size=2048,
        packing_algorithm="first_fit_decreasing",  # Can also use "first_fit_decreasing"
        seed=42,
        num_workers=4
    )

    dataprocess.inspect_dataset(input_dir="packed_ffd/train", tokenizer="EuroBERT/EuroBERT-210m", num_samples=2, tiktoken=False)
    
    print("\nFirst-Fit Decreasing Metadata:")
    with open("packed_ffd/packing_metadata.json", "r") as f:
        print(json.dumps(json.load(f), indent=2))

    # ## 4. Create the Datamix

    # ### Creating the Data Mix
    # 
    # Now that our data is processed, we can create the data mix. This JSON file lists the different datasets we've processed and want to include during training. We can choose how many samples to take from each dataset, and the Optimus library will automatically create the training mix, ensuring the data is shuffled across all datasets.
    # 
    # - Proportion: ratio (float)
    # - Choose: samples (int)
    # 
    # ```json
    # [
    #   {
    #     "local": "dataset_processed_path",
    #     "choose": 200,
    #   },
    #     {
    #     "local": "dataset2_processed_path",
    #     "proportion": 1.5,
    #   }
    # ]
    # ```

    import os

    # Use the most efficient packed dataset (first_fit_shuffle)
    train = [
        {
            "local": "packed_ffs/train",
            "choose": 200
        },
    ]

    os.makedirs("datamix", exist_ok=True)
    with open("datamix/train.json", "w") as f:
        json.dump(train, f)

    # **Make sure the mix is named `train.json`, or Optimus won't be able to find it during training.**

    # ---

    # ## 5. Training

    # Now that our data is processed, we can start training our model. In this section, we'll use Python entirely. You can also run the command `python optimus.train` with all the configuration arguments, which will give you similar results. For example:
    # ```bash
    # python -m optimus.train --huggingface_id EuroBERT/EuroBERT-210m --output_dir "/content/model" --lr_scheduler "OneCycleLR" --div_factor 10 --end_start 0.9 --final_div_factor 100 --save_step 100 --data_mix_path "/content/datamix" --batch_size 1 --gpu
    # ```

    #from optimus.trainer.configuration.configs import Config
    #from optimus.trainer.data import Data
    #from optimus.trainer.model.load import load_model, load_tokenizer
    #from optimus.trainer.pretrain import Pretrain

    # Let's configure our training! Here, we specify the model name, learning rate settings, and the data mix.

    # config = Config()
    #
    # config.model.huggingface_id = "EuroBERT/EuroBERT-210m"
    # config.model.gpu = True # If you don't have GPU set it to False.
    #
    # config.train.output_dir = "model"
    # config.train.lr_scheduler = "OneCycleLR"
    # config.train.div_factor = 10
    # config.train.pct_start = 0.3
    # config.train.final_div_factor = 100
    # config.train.save_step = 100
    #
    # config.data.data_mix_path = "datamix"
    # config.data.batch_size = 1

    # We recommend checking out the [training documentation](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/docs/trainer.md) in Optimus for the full list of configuration options. Alternatively, you can run the following Python code to print the different configuration sections:
    # 
    # ```python
    # print("Model")
    # print(json.dumps(asdict(config.model), indent=4))
    # print("Data")
    # print(json.dumps(asdict(config.data), indent=4))
    # print("Train")
    # print(json.dumps(asdict(config.train), indent=4))
    # ```

    # Now, we can launch the training! In this example, we're not using distributed training, so we set the distributed object responsible for training supervision to `None`.

    #distributed = None
    #
    #model = load_model(config)
    #tokenizer = load_tokenizer(config)
    #
    #data = Data(config, tokenizer)
    #
    #pretrain = Pretrain(model, data, distributed, config)
    #
    #print(data)
    #print(pretrain)
