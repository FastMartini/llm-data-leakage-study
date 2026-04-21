# Membership Inference Attack (MIA) Project Guide

## How to run

### 1. Prerequisites

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

---

### 2. Running the Program

### Step 1: Train the Target Model
```bash
python3 train.py
```

### Step 2: Run the Attack
```bash
python3 attack.py
```

---

## Overview

This project demonstrates a **membership inference attack (MIA)** on a machine learning model trained on text data.

A membership inference attack asks a very specific privacy question:

> Given a data sample and the model’s output on that sample, can we infer whether that sample was part of the model’s training set?

That question matters because a model may reveal private information **without directly revealing the training data itself**. If an attacker can determine that a specific person’s record was used in training, that can already be a privacy leak.

This guide is written for beginners. It explains:
- what **membership** means
- what each Python file does
- what each major block of code is doing
- why the project is structured this way
- how to talk about the project clearly in a presentation

---

# What “Membership” Means

In this project, **membership** means:

> whether a specific data point was used during the training of the target model

That gives every sample a binary label:

- `1` = **member**
- `0` = **non-member**


---

# Why Membership Matters

At first, this may seem like a small detail. But in real privacy-sensitive applications, membership can reveal important facts.

For example:

- If a model was trained on hospital records, membership might reveal whether a person’s medical record was included.
- If a model was trained on private emails, membership might reveal whether someone’s email was used.
- If a model was trained on legal or financial documents, membership might reveal whether a specific document was part of training.

That is why membership inference is considered a **privacy attack**.

The model does not need to reveal the full text of the private record for privacy harm to happen. Simply revealing whether that record was included can already be sensitive.

---

# Simple Analogy

Imagine a student studies 500 flashcards before a quiz.

Later, you show the student 1,000 flashcards:
- 500 that they studied
- 500 that they never saw

If the student answers some cards with much higher confidence, you may guess:

> “That was probably part of the study set.”

That is exactly the idea behind membership inference.

Translate the analogy like this:

- **Student** = trained model
- **Studied flashcards** = training samples
- **New flashcards** = unseen samples
- **Confidence of answer** = attack signal

The attacker is trying to detect whether the model behaves differently on things it has seen before.

---

# Project Structure

This project is organized into three files:

1. `dataset.py`
2. `train.py`
3. `attack.py`

Each file has a specific purpose:

- `dataset.py` prepares the dataset
- `train.py` trains the target model
- `attack.py` performs the privacy attack

This is a good design because each file has one main job. That makes the code easier to understand, test, explain, and present.

---

# Full Pipeline

The experiment works like this:

1. Load the IMDB dataset
2. Create two groups:
   - members
   - non-members
3. Convert text into numbers using TF-IDF
4. Train a sentiment classifier on members only
5. Ask the classifier for probability outputs on both groups
6. Extract behavior-based attack signals
7. Use those signals to infer membership

So this project really involves **two tasks**:

## Task 1: Target model task
Predict sentiment:
- positive
- negative

## Task 2: Attack model task
Predict membership:
- member
- non-member

That is what makes the project more sophisticated than ordinary text classification.

---

# Why the IMDB Dataset Was Chosen

The IMDB dataset is a good fit because:

- it is text-based
- it already has labels
- it is easy to explain
- it comes with train/test splits
- it works well with standard text-classification tools
- it is large enough to sample from

Each IMDB example contains:
- review text
- sentiment label

The sentiment label is:
- `0` = negative
- `1` = positive

This gives the project a clean classification task while still allowing you to study privacy leakage.

---

# File 1: `dataset.py`

## Purpose

`dataset.py` is responsible for preparing the data.

Its job is to decide:
- which samples become members
- which samples become non-members

Without that split, the rest of the project cannot exist.

---

## Code

```python
# Why: load_dataset downloads and loads the IMDB dataset in a structured format.
from datasets import load_dataset

# Why: train_test_split helps us create reproducible member and non-member subsets.
from sklearn.model_selection import train_test_split


# Why: this function centralizes dataset loading and splitting so the rest of the project can reuse it.
def load_data(member_size=500, non_member_size=500, random_state=42):
    # Why: this loads the full IMDB dataset, including official train and test splits.
    dataset = load_dataset("imdb")

    # Why: these are the full text and label lists from the official training split.
    train_texts_full = dataset["train"]["text"]
    train_labels_full = dataset["train"]["label"]

    # Why: these are the full text and label lists from the official test split.
    test_texts_full = dataset["test"]["text"]
    test_labels_full = dataset["test"]["label"]

    # Why: members must come from the training split because the target model will see them during training.
    member_texts, _, member_labels, _ = train_test_split(
        train_texts_full,
        train_labels_full,
        train_size=member_size,
        stratify=train_labels_full,
        random_state=random_state,
    )

    # Why: non-members must come from the test split so they remain unseen by the target model.
    non_member_texts, _, non_member_labels, _ = train_test_split(
        test_texts_full,
        test_labels_full,
        train_size=non_member_size,
        stratify=test_labels_full,
        random_state=random_state,
    )

    # Why: returning these four objects gives downstream files exactly what they need.
    return member_texts, member_labels, non_member_texts, non_member_labels


# Why: this block lets you test dataset loading directly without affecting imports in other files.
if __name__ == "__main__":
    # Why: load the default controlled experiment split for a sanity check.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: these prints confirm the split sizes are correct.
    print("Members:", len(member_texts))
    print("Non-members:", len(non_member_texts))

    # Why: these prints help verify that the data looks valid.
    print("First member label:", member_labels[0])
    print("First member review preview:", member_texts[0][:200])
```

---

## Detailed Explanation

### `from datasets import load_dataset`

This imports the Hugging Face dataset loader.

Why this matters:
- it saves you from downloading files manually
- it keeps the experiment reproducible
- it makes the setup cleaner for demonstrations

---

### `from sklearn.model_selection import train_test_split`

This imports a function that can split or sample data.

In this project, it is used to create smaller controlled subsets:
- 500 members
- 500 non-members

This keeps the experiment manageable and easier to explain.

---

### `def load_data(...)`

This function wraps all of the dataset logic into one reusable place.

That means:
- `train.py` can reuse it
- `attack.py` can indirectly reuse it through `train.py`

This is better than copying the same code into multiple files.

---

### `dataset = load_dataset("imdb")`

This downloads and loads the IMDB dataset.

The IMDB dataset already includes:
- `dataset["train"]`
- `dataset["test"]`

This is useful because it gives you a natural way to define:
- members from train
- non-members from test

---

### Extracting texts and labels

```python
train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]

test_texts_full = dataset["test"]["text"]
test_labels_full = dataset["test"]["label"]
```

Machine learning usually separates:
- **inputs** = the data the model reads
- **labels** = the correct answers

Here:
- inputs = review text
- labels = sentiment

---

### Creating the member set

```python
member_texts, _, member_labels, _ = train_test_split(...)
```

The member set is sampled from the training split.

These samples are members because the target model will actually train on them later.

That is why the word **membership** is meaningful here.

---

### `stratify=train_labels_full`

This preserves label balance.

That means your smaller subset should still have a reasonable balance of:
- positive reviews
- negative reviews

This helps keep the experiment fair and stable.

---

### Creating the non-member set

```python
non_member_texts, _, non_member_labels, _ = train_test_split(...)
```

The non-member set is sampled from the official test split.

These reviews are never shown to the model during training.

That makes them valid non-members.

---

### `return ...`

This returns the four objects needed by the next stage:
- member texts
- member labels
- non-member texts
- non-member labels

---

### `if __name__ == "__main__":`

This lets the file behave in two ways:
- as a script when run directly
- as a module when imported elsewhere

If you run:

```bash
python dataset.py
```

it will print:
- how many members you have
- how many non-members you have
- a small review preview

That is helpful for checking whether the data stage works.

---

# File 2: `train.py`

## Purpose

`train.py` trains the **target model**.

The target model is the model that will later be attacked.

In this project, the target model learns sentiment classification:
- positive review
- negative review

But from the privacy side, what really matters is this:

> Does the target model behave differently on training samples than on unseen samples?

If the answer is yes, that difference may create privacy leakage.

---

## Why text must be converted into numbers

A model cannot directly understand raw text.

For example, a model cannot directly process:

> "This movie was fantastic."

It needs numerical input.

That is why `train.py` uses **TF-IDF**.

---

## What TF-IDF does

TF-IDF stands for:
- Term Frequency
- Inverse Document Frequency

It converts each review into a vector of numbers based on the importance of words.

General intuition:
- words that appear frequently in one review may matter
- words that appear in almost every review may matter less

This gives the model a numeric representation of language.

---

## Code

```python
# Why: load_data gives us the prepared member and non-member groups from dataset.py.
from dataset import load_data

# Why: TfidfVectorizer converts raw text into numerical feature vectors the model can learn from.
from sklearn.feature_extraction.text import TfidfVectorizer

# Why: LogisticRegression is a simple and effective baseline model for text classification.
from sklearn.linear_model import LogisticRegression


# Why: this function handles the full target-model training workflow and returns everything needed later.
def train_target_model(member_size=500, non_member_size=500, random_state=42):
    # Why: load the controlled experiment data so the target model only trains on members.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data(
        member_size=member_size,
        non_member_size=non_member_size,
        random_state=random_state,
    )

    # Why: the vectorizer learns a vocabulary from member data only, which avoids leakage from unseen data.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    # Why: fit_transform learns the vocabulary from member text and converts it into feature vectors.
    X_member = vectorizer.fit_transform(member_texts)

    # Why: transform applies the same learned vocabulary to non-member text without refitting.
    X_non_member = vectorizer.transform(non_member_texts)

    # Why: the target model is the classifier we will later attack.
    model = LogisticRegression(max_iter=1000, random_state=random_state)

    # Why: the model must only train on members for the membership labels to be meaningful.
    model.fit(X_member, member_labels)

    # Why: these scores show how differently the model behaves on seen versus unseen data.
    member_accuracy = model.score(X_member, member_labels)
    non_member_accuracy = model.score(X_non_member, non_member_labels)

    # Why: returning a dictionary keeps the code organized and makes attack.py easier to write.
    return {
        "model": model,
        "vectorizer": vectorizer,
        "member_texts": member_texts,
        "member_labels": member_labels,
        "non_member_texts": non_member_texts,
        "non_member_labels": non_member_labels,
        "X_member": X_member,
        "X_non_member": X_non_member,
        "member_accuracy": member_accuracy,
        "non_member_accuracy": non_member_accuracy,
    }


# Why: this block lets you run train.py directly to verify target-model training.
if __name__ == "__main__":
    # Why: train the model and capture all relevant outputs.
    results = train_target_model()

    # Why: print these values so you can inspect whether the target model behaves differently on members and non-members.
    print("Member accuracy:", results["member_accuracy"])
    print("Non-member accuracy:", results["non_member_accuracy"])
```

---

## Detailed Explanation

### `from dataset import load_data`

This imports the data-preparation function from `dataset.py`.

This is good project organization because:
- one file handles data
- one file handles training

---

### `from sklearn.feature_extraction.text import TfidfVectorizer`

This imports the text vectorizer.

Its job is to translate text into numbers.

Without this, the target model would have no numerical features to learn from.

---

### `from sklearn.linear_model import LogisticRegression`

This imports the target classifier.

Even though the name includes “regression,” logistic regression is widely used for classification tasks.

It is a strong baseline because it is:
- simple
- fast
- standard
- easy to explain in a presentation

---

### `member_texts, ... = load_data(...)`

This loads the member and non-member sets.

At this point:
- members are training data
- non-members are unseen data

That split is what gives the attack meaning.

---

### `vectorizer = TfidfVectorizer(...)`

This creates the TF-IDF processor.

#### `max_features=5000`
Keeps at most 5,000 vocabulary terms.

This helps:
- control model size
- keep memory use reasonable
- simplify the experiment

#### `stop_words="english"`
Removes very common words such as:
- the
- and
- is
- of

These words usually add little sentiment information.

---

### `X_member = vectorizer.fit_transform(member_texts)`

This line does two things:
1. learns the vocabulary from member data
2. converts member text into vectors

This is important because vocabulary learning should come only from training data.

---

### `X_non_member = vectorizer.transform(non_member_texts)`

This applies the already-learned vocabulary to non-member text.

Notice that it does **not** use `fit_transform`.

That is intentional.

Why:
- non-members should not influence the vocabulary
- this keeps the setup honest

---

### `model = LogisticRegression(...)`

This creates the target model.

#### `max_iter=1000`
Gives the optimizer enough iterations to converge.

#### `random_state=random_state`
Helps keep the run reproducible.

---

### `model.fit(X_member, member_labels)`

This trains the target model on members only.

That is why the membership labels remain meaningful.

If you accidentally trained on both members and non-members, the privacy experiment would break.

---

### `member_accuracy = model.score(...)`

This checks how accurate the model is on the same data it trained on.

This is usually high.

---

### `non_member_accuracy = model.score(...)`

This checks how accurate the model is on unseen data.

This is often lower than member accuracy.

That difference can reflect overfitting or memorization, which may increase privacy leakage.

---

# File 3: `attack.py`

## Purpose

`attack.py` performs the membership inference attack.

Its job is to use the target model’s behavior to predict whether a sample was:
- a member
- a non-member

This is the privacy part of the project.

---

## Main idea

The target model never explicitly says:

- “yes, I trained on this review”
- “no, I did not train on this review”

So the attacker must infer membership indirectly.

The attacker uses behavior-based signals such as:
- confidence
- true-class confidence
- loss
- entropy
- correctness

These signals can differ between:
- samples seen during training
- samples never seen during training

---

## What the attack features mean

### Max confidence
How certain the model is about its chosen label.

### True-class confidence
How much probability the model assigns to the correct label.

### Loss
How well the model fits the true label.

### Entropy
How uncertain the model is overall.

### Correctness
Whether the model predicted the label correctly.

These signals together create an attack dataset.

---

## Code

```python
# Why: this gives attack.py access to the trained target model and its feature matrices.
from train import train_target_model

# Why: numpy is used for efficient numeric operations on model outputs and derived attack features.
import numpy as np

# Why: pandas makes it easier to organize attack features into a readable table-like structure.
import pandas as pd

# Why: train_test_split is used to create a train/test split for the attack model itself.
from sklearn.model_selection import train_test_split

# Why: LogisticRegression is used here again as a simple attack model that predicts member vs non-member.
from sklearn.linear_model import LogisticRegression

# Why: these metrics provide a stronger evaluation than plain accuracy alone.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Why: this helper computes Shannon entropy, which measures how uncertain the model is about a prediction.
def compute_entropy(probabilities):
    # Why: clipping prevents log(0), which would be undefined and break the calculation.
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)

    # Why: entropy is lower when the model is very certain and higher when the model is more uncertain.
    return -np.sum(clipped_probabilities * np.log(clipped_probabilities), axis=1)


# Why: this helper computes the negative log probability of the true class, which is the sample loss.
def compute_true_class_loss(probabilities, true_labels):
    # Why: clipping prevents log(0) and keeps the loss calculation numerically stable.
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)

    # Why: this selects the predicted probability assigned to the true class for each sample.
    true_class_probabilities = clipped_probabilities[np.arange(len(true_labels)), true_labels]

    # Why: lower loss usually indicates the model handled that sample more confidently and correctly.
    return -np.log(true_class_probabilities)


# Why: this helper converts target-model behavior into attack features for one set of samples.
def build_attack_features(model, feature_matrix, true_labels, membership_label):
    # Why: predict_proba is the core signal source because it gives probability behavior for each sample.
    probabilities = model.predict_proba(feature_matrix)

    # Why: predict returns the final class label so we can see whether the target model was correct.
    predictions = model.predict(feature_matrix)

    # Why: max confidence is a common membership signal because training samples often receive stronger confidence.
    max_confidence = np.max(probabilities, axis=1)

    # Why: true-class confidence measures how much probability the model assigned to the correct label specifically.
    true_class_confidence = probabilities[np.arange(len(true_labels)), true_labels]

    # Why: loss is often stronger than plain confidence because it directly measures how well the model fits a sample.
    true_class_loss = compute_true_class_loss(probabilities, true_labels)

    # Why: entropy captures uncertainty and can reveal whether the model is more certain on members.
    entropy = compute_entropy(probabilities)

    # Why: correctness is a useful binary signal because members may be predicted correctly more often.
    correctness = (predictions == np.array(true_labels)).astype(int)

    # Why: every attack row must carry the true membership label so the attack model can learn from it.
    membership = np.full(len(true_labels), membership_label)

    # Why: packaging attack signals into a DataFrame makes the attack dataset easy to inspect and use.
    attack_df = pd.DataFrame(
        {
            "max_confidence": max_confidence,
            "true_class_confidence": true_class_confidence,
            "loss": true_class_loss,
            "entropy": entropy,
            "correctness": correctness,
            "membership": membership,
        }
    )

    return attack_df
```

---

## Detailed Explanation

### `from train import train_target_model`

This imports the trained target-model pipeline.

That means `attack.py` does not have to repeat training logic.

---

### Why NumPy is used

`numpy` handles:
- arrays
- math
- indexing
- probability-based calculations

It is especially useful for:
- entropy
- loss
- combining attack signals

---

### Why pandas is used

`pandas` makes the attack features easier to organize and inspect.

Instead of handling loose arrays everywhere, you can create a table with columns like:
- max confidence
- loss
- entropy
- correctness
- membership

That makes the attack model cleaner and easier to explain.

---

### `compute_entropy(probabilities)`

Entropy measures uncertainty.

Examples:
- `[0.99, 0.01]` → low entropy, very certain
- `[0.50, 0.50]` → high entropy, very uncertain

Models are often more certain on training data, so entropy can be a useful membership signal.

---

### `compute_true_class_loss(probabilities, true_labels)`

Loss measures how well the model handled the true label.

If the model assigns a high probability to the correct class, loss is low.

If it assigns a low probability to the correct class, loss is high.

Members often have lower loss than non-members.

---

### `build_attack_features(...)`

This function turns target-model behavior into an attack dataset.

For each sample, it computes:
- max confidence
- true-class confidence
- loss
- entropy
- correctness
- membership label

This is a key step because it transforms the privacy problem into a second machine-learning problem.

Now the attack model can learn from these behavior signals.

---

# Threshold Attack vs Learned Attack

A strong way to present your project is to explain that it contains **two attack styles**.

## 1. Threshold attack
A simple rule:
- if confidence is above a threshold, guess member
- otherwise, guess non-member

This is easy to understand and useful as a baseline.

## 2. Learned attack
A second classifier is trained on attack features.

That classifier learns patterns in:
- confidence
- loss
- entropy
- correctness

This is more sophisticated and more presentation-worthy.

---



