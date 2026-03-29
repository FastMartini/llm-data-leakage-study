# IMDB Dataset Setup for Membership Inference Attack

## Overview

This script prepares the dataset for a controlled **membership inference attack (MIA)** experiment.

A membership inference attack tries to determine whether a specific data sample was part of a model's training data. To test that properly, we need two clearly separated groups:

- **Members**: samples the model will train on
- **Non-members**: samples the model will not see during training

This script uses the **IMDB movie review dataset** from Hugging Face and creates those two groups.

---

## Code Explanation

### 1. Import the required libraries

```python
# Why: this library lets you directly download datasets from Hugging Face
from datasets import load_dataset

# Why: train_test_split helps us create a smaller, controlled experiment.
from sklearn.model_selection import train_test_split

```

### What this does
- `load_dataset` is used to download and load the IMDB dataset.
- `train_test_split` is used to sample smaller subsets from the full dataset

The full IMDB dataset is large. For an initial MIA experiment, a smaller controlled subset is easier to work with and easier to analyze.

### 2. Load the IMDB dataset

```python
# Why: this line downloads and loads the IMDB dataset into memory
dataset = load_dataset("imdb")
```

### What this does

This downloads the IMDB movie review dataset and loads it into Python.

### What the dataset contains

The IMDB dataset comes with two official splits:
- `train`: 25,000 movie reviews
- `test`: 25,000 movie reviews

Each review has:
- `text`: the review itself
- `label`: sentiment label
    - `0` = negative
    - `1` = positive

We need text samples and labels so we can later train a sentiment classifier.

### 3. Inspect the dataset:

```python
# Why: confirms dataset structure and splits
print(dataset)

# Why: shows one sample so you understand the format
print(dataset["train"][0])
```

### What this does
- `print(dataset)` shows the dataset structure and available splits.
- `print(dataset["train"][0])` prints one example review and its label.

### 4. Extract the full text and labels

```python
# Why: we separate texts and labels because the model needs inputs and targets independently.
train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]

test_texts_full = dataset["test"]["text"]
test_labels_full = dataset["test"]["label"]
```

### What this does

This pulls out:
- all training texts
- all training labels
- all test texts
- all test labels

### 5. Create the member set

```python
# Why: these are the member samples the model will train on.
member_texts, _, member_labels, _ = train_test_split(
    train_texts_full,
    train_labels_full,
    train_size=500,
    stratify=train_labels_full,
    random_state=42
)
```

### What this does
This selects 500 samples from the original training split. These 500 samples become the member set. These are the samples the target model will eventually train on. In the MIA experiment, these are the reviews the model has "seen."

- `train_size=500`: selects exactly 500 samples
- `stratify=train_labels_full`: preserves the class balance, so the subset still has a reasonable mix of positive and negative reviews
- `random_state=42`: makes the split reproducible, meaning the same 500 samples are selected each time the code is run

### 6. Create the non-member set

```python
# Why: these are the non-member samples the model will never see during training.
non_member_texts, _, non_member_labels, _ = train_test_split(
    test_texts_full,
    test_labels_full,
    train_size=500,
    stratify=test_labels_full,
    random_state=42
)
```

### What this does
This selects 500 samples from the official IMDB test split. These 500 samples become the non-member set. These are the samples the target model should never train on. In the MIA experiment, these act as the "unseen" examples.

### 7. Verify the result

```python
# Why: these prints verify the sizes of the two groups.
print("Members:", len(member_texts))
print("Non-members:", len(non_member_texts))
```

### What this does
This prints the size of each group.

Expected output:

```python
Members: 500
Non-members: 500
```




