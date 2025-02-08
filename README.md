#  Fine-Tuning DistilBERT with LoRA for Sentiment Classification

## Overview
This project demonstrates fine-tuning a **DistilBERT** model with **Low-Rank Adaptation (LoRA)** for binary sentiment classification on the IMDb dataset. The fine-tuned model is then uploaded to **Hugging Face Hub** for inference.

## Features
- Uses **Hugging Face Transformers** for model training and tokenization.
- Applies **LoRA** for parameter-efficient fine-tuning.
- Evaluates model performance using accuracy.
- Saves and uploads the trained model to **Hugging Face Hub**.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install datasets transformers peft evaluate torch numpy
```

## Dataset Preparation
- The IMDb dataset is loaded from Hugging Face's `datasets` library.
- A **subset** of 1000 samples is randomly selected for training and evaluation.
- The dataset is formatted into a `DatasetDict` with `train` and `validation` splits.

```python
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np

dataset = load_dataset("shawhin/imdb-truncated")
print(dataset)
```

## Model & Tokenizer Setup
- Uses **DistilBERT** (`distilbert-base-uncased`), but **RoBERTa** can be used as an alternative.
- Adds a **classification head** with two labels: `Negative` and `Positive`.
- Applies tokenization and padding.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

## LoRA Configuration
- **LoRA (Low-Rank Adaptation)** is used for efficient fine-tuning.
- Only a small subset of model parameters is updated.

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## Training Setup
- Defines a **Trainer** with evaluation and saving strategies.
- Uses a **data collator** to handle padding.

```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate

training_args = TrainingArguments(
    output_dir="distilbert-lora-imdb",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```

## Model Evaluation
- Evaluates sentiment classification on sample texts before and after training.

```python
import torch
text_list = ["It was good.", "Not a fan, don’t recommend."]
model.to('cpu')
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt")
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(f"{text} - {id2label[predictions.item()]}")
```

## Upload Model to Hugging Face Hub
Ensure you are logged in to **Hugging Face Hub**:
```python
from huggingface_hub import notebook_login
notebook_login()
```

Push the trained model and tokenizer:
```python
hf_name = 'shawhin'
model_id = f"{hf_name}/distilbert-lora-imdb"
model.push_to_hub(model_id)
trainer.push_to_hub(model_id)
```

## Load Model for Inference
To reload the model from the **Hugging Face Hub**:
```python
from peft import PeftConfig, PeftModel
config = PeftConfig.from_pretrained(model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id
)
model = PeftModel.from_pretrained(inference_model, model_id)
```



