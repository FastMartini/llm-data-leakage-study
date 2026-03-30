# Membership Inference Attack (MIA)

## Overview

This project demonstrates a **membership inference attack (MIA)** on a machine learning model trained on text data.

A membership inference attack tries to answer the following question:

> Given a piece of data and the model's behavior on that data, can we guess whether that piece of data was used during training?

That is the core privacy question behind this experiment.

This project is organized into three Python files:

1. `dataset.py`
2. `train.py`
3. `attack.py`

Each file has a specific role:

- `dataset.py` prepares the data
- `train.py` builds and trains the target model
- `attack.py` tries to infer which samples were part of training

This markdown file explains:
- the purpose of each file
- the purpose of each major function
- the meaning of the important machine learning concepts
- how the entire pipeline works from beginning to end
- why each design decision matters

---

# What Is a Membership Inference Attack?

Before discussing the code, it is important to understand the idea behind the project.

Imagine a student studies 500 flashcards before a quiz. Then you show the student 1,000 flashcards:
- 500 they studied before
- 500 they have never seen

Now you try to guess which flashcards they studied by looking at how confidently they answer.

If they respond very quickly and confidently to the studied flashcards, and more cautiously to the new ones, then you might be able to tell which flashcards were part of their study set.

That is very similar to what a membership inference attack does.

In machine learning terms:

- The **student** is the trained model
- The **flashcards it studied** are the training samples
- The **new flashcards** are unseen samples
- The **confidence of the answers** becomes the signal that the attacker uses

The privacy concern is that a model may accidentally reveal whether a certain record was used during training.

That matters because the training data may contain private information such as:
- medical records
- financial data
- private emails
- private conversations
- personal text documents

If an attacker can determine that a certain person's data was part of training, that can already be a privacy leak, even if the model does not reveal the full contents directly.

---

# High-Level Project Flow

The full experiment works like this:

1. Load a labeled dataset of text
2. Split it into:
   - **members**: data the model will train on
   - **non-members**: data the model will never see during training
3. Convert text into numbers using TF-IDF
4. Train a target model on the member set only
5. Ask the target model for confidence scores on both groups
6. Compare confidence on members versus non-members
7. Use that difference to perform a membership inference attack

In this project, the dataset is the **IMDB movie review dataset**.

Each review has:
- review text
- sentiment label

The sentiment label is:
- `0` = negative review
- `1` = positive review

---

# Why IMDB Was Chosen

The IMDB dataset is a strong choice for a controlled experiment because:

- it is a clean text classification dataset
- it already comes with labels
- it has separate training and test splits
- it is easy to explain
- it is large enough to sample from
- it works well with baseline models like logistic regression

This makes it a very good dataset for demonstrating privacy leakage in a simple, controlled setting.

---

# File 1: `dataset.py`

## Purpose of `dataset.py`

This file is responsible for **loading the IMDB dataset and creating the two groups needed for the attack**.

Those two groups are:

- **members**: data the model will train on
- **non-members**: data the model will not train on

Without these two groups, there is no membership inference experiment.

---

## Core Idea of `dataset.py`

The main job of this file is to answer:

> Which reviews belong to training, and which reviews stay unseen?


---

## Code

```python
# Why: load_dataset downloads and loads the IMDB dataset from Hugging Face.
from datasets import load_dataset

# Why: train_test_split lets us sample smaller, reproducible member and non-member subsets.
from sklearn.model_selection import train_test_split


# Why: putting the dataset logic in a function makes it reusable from train.py and attack.py.
def load_data(member_size=500, non_member_size=500, random_state=42):
    # Why: this downloads and loads the full IMDB dataset into memory.
    dataset = load_dataset("imdb")

    # Why: these are the full official train texts and labels that we will sample members from.
    train_texts_full = dataset["train"]["text"]
    train_labels_full = dataset["train"]["label"]

    # Why: these are the full official test texts and labels that we will sample non-members from.
    test_texts_full = dataset["test"]["text"]
    test_labels_full = dataset["test"]["label"]

    # Why: members are sampled from the training split because the target model will train on them.
    member_texts, _, member_labels, _ = train_test_split(
        train_texts_full,
        train_labels_full,
        train_size=member_size,
        stratify=train_labels_full,
        random_state=random_state,
    )

    # Why: non-members are sampled from the test split so they remain unseen during training.
    non_member_texts, _, non_member_labels, _ = train_test_split(
        test_texts_full,
        test_labels_full,
        train_size=non_member_size,
        stratify=test_labels_full,
        random_state=random_state,
    )

    # Why: returning these four objects makes the dataset easy to reuse elsewhere.
    return member_texts, member_labels, non_member_texts, non_member_labels


# Why: this block lets you run dataset.py directly for a quick sanity check without affecting imports.
if __name__ == "__main__":
    # Why: load the default dataset split so we can verify everything is working.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: these prints confirm the two groups were created with the expected sizes.
    print("Members:", len(member_texts))
    print("Non-members:", len(non_member_texts))

    # Why: these prints help verify that labels exist and the text looks correct.
    print("First member label:", member_labels[0])
    print("First member review preview:", member_texts[0][:200])
```

---

## Step-by-Step Explanation of `dataset.py`

### 1. Importing `load_dataset`

```python
from datasets import load_dataset
```

This import comes from the Hugging Face `datasets` library.

Its purpose is to make it easy to download and load prepared datasets.

Instead of manually:
- downloading a CSV file
- unzipping it
- parsing it
- organizing train/test splits

you can use a single function call:

```python
load_dataset("imdb")
```

That makes the project much cleaner and easier to reproduce.

---

### 2. Importing `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

This function is commonly used in machine learning to split data into different groups.

In this project, it is used to create smaller subsets:
- 500 member reviews
- 500 non-member reviews

We do not use the full 25,000 reviews in each split because this is a controlled experiment and smaller subsets are easier to:
- run quickly
- inspect
- debug
- explain
- overfit slightly if needed

---

### 3. The `load_data()` function

```python
def load_data(member_size=500, non_member_size=500, random_state=42):
```

This function wraps the dataset logic into one reusable unit.

That means:
- `train.py` can import and reuse it
- `attack.py` can reuse it indirectly through `train.py`

This is better than writing the same logic multiple times.

#### Parameters

- `member_size=500`
  - number of training samples to use as members
- `non_member_size=500`
  - number of unseen samples to use as non-members
- `random_state=42`
  - makes the random sampling reproducible

### Why reproducibility matters

If you run a random split without fixing the random seed, you may get different samples every time.

That makes results harder to:
- compare
- debug
- report

Using `random_state=42` means the same subset is chosen each time, assuming the dataset stays the same.

---

### 4. Loading the full dataset

```python
dataset = load_dataset("imdb")
```

This downloads and loads the IMDB dataset.

The dataset has two official parts:

- `dataset["train"]`
- `dataset["test"]`

Each item in the dataset looks like:

```python
{
    "text": "This movie was fantastic ...",
    "label": 1
}
```

So each review contains:
- the text of the review
- the sentiment label

---

### 5. Extracting texts and labels

```python
train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]

test_texts_full = dataset["test"]["text"]
test_labels_full = dataset["test"]["label"]
```

This separates the raw inputs from the labels.

This is a common pattern in machine learning:

- **inputs** are what the model reads
- **labels** are the correct answers the model tries to learn

In this case:
- inputs = movie review text
- labels = positive or negative sentiment

---

### 6. Creating the member set

```python
member_texts, _, member_labels, _ = train_test_split(
    train_texts_full,
    train_labels_full,
    train_size=member_size,
    stratify=train_labels_full,
    random_state=random_state,
)
```

This samples the member set from the official training split.

These samples are called **members** because the model will later train on them.

#### Why use training data for members?

Because the model must truly see these samples during training.

If a sample is supposed to be a member, it must actually be part of the training process.

---

### 7. Understanding `stratify=train_labels_full`

This is very important.

`stratify=train_labels_full` tells the splitting function to preserve the label balance.

That means if the original data is roughly balanced between:
- positive reviews
- negative reviews

then the sampled subset will also be roughly balanced.

This matters because a badly imbalanced subset can distort:
- model training
- confidence behavior
- attack results

---

### 8. Creating the non-member set

```python
non_member_texts, _, non_member_labels, _ = train_test_split(
    test_texts_full,
    test_labels_full,
    train_size=non_member_size,
    stratify=test_labels_full,
    random_state=random_state,
)
```

This samples the non-member set from the official test split.

These reviews are never shown to the model during training.

That is why they are valid non-members.

#### Why sample from the official test split?

Because the official test split is already separate from the training split.

That makes the setup clean and easy to defend:
- no overlap
- no accidental leakage
- clear experiment design

---

### 9. Returning the four objects

```python
return member_texts, member_labels, non_member_texts, non_member_labels
```

These four outputs are exactly what the next stage needs.

They represent:
- member inputs
- member labels
- non-member inputs
- non-member labels

---

### 10. The `if __name__ == "__main__":` block

```python
if __name__ == "__main__":
```

This line is often confusing for beginners.

Here is the simple explanation:

- If you run `python dataset.py` directly, this block runs.
- If another file imports `dataset.py`, this block does not run.

That lets the file serve two roles:
- a reusable module
- a standalone test script

This is very helpful because you can:
- import `load_data()` into `train.py`
- still run `dataset.py` by itself to check that the data setup is working

---

## What You Should Expect When Running `dataset.py`

When you run:

```bash
python dataset.py
```

you should see:
- `Members: 500`
- `Non-members: 500`
- the label of the first member
- a preview of the first review

That confirms the dataset preparation stage is working.

---

# File 2: `train.py`

## Purpose of `train.py`

This file trains the **target model**.

In a membership inference attack project, the target model is the model being attacked.

So the point of `train.py` is to:
- receive the member and non-member data
- convert the text into numerical features
- train a classifier on members only
- report how the model performs

---

## Why Text Must Be Converted Into Numbers

Machine learning models cannot understand raw text directly.

For example, a model cannot directly process:

```text
"This movie was amazing and emotional."
```

It needs numbers, not plain sentences.

That is why we use **TF-IDF**.

TF-IDF turns text into numerical vectors.

---

## What Is TF-IDF?

TF-IDF stands for:

- **Term Frequency**
- **Inverse Document Frequency**

The basic idea is:

- words that appear often in one review may be important
- words that appear in almost every review may be less important

For example:
- a word like `"movie"` appears in many reviews, so it is not very special
- a word like `"masterpiece"` may be more informative for sentiment

TF-IDF assigns weights to words based on this idea.

So each review becomes a vector of numbers representing the importance of words.

---

## Code

```python
# Why: we import the prepared member and non-member data from dataset.py instead of duplicating data logic.
from dataset import load_data

# Why: TfidfVectorizer converts raw review text into numerical features the classifier can learn from.
from sklearn.feature_extraction.text import TfidfVectorizer

# Why: LogisticRegression is a simple, fast baseline model for text classification.
from sklearn.linear_model import LogisticRegression


# Why: wrapping the training workflow in a function makes it reusable from attack.py.
def train_target_model():
    # Why: load the member and non-member sets for the controlled experiment.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: the vectorizer learns a vocabulary from member data only, which matches a real training setup.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    # Why: fit_transform learns the vocabulary on training members and converts them into feature vectors.
    X_member = vectorizer.fit_transform(member_texts)

    # Why: transform applies that same vocabulary to unseen non-member reviews.
    X_non_member = vectorizer.transform(non_member_texts)

    # Why: this is the target model that the membership inference attack will later probe.
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Why: the model must train only on member data so those samples truly count as members.
    model.fit(X_member, member_labels)

    # Why: these scores give a quick sanity check before moving to the attack stage.
    member_accuracy = model.score(X_member, member_labels)
    non_member_accuracy = model.score(X_non_member, non_member_labels)

    # Why: returning everything needed later keeps attack.py simple.
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


# Why: this block lets you run train.py directly to verify the target model trains correctly.
if __name__ == "__main__":
    # Why: train the model and capture all returned experiment objects.
    results = train_target_model()

    # Why: print accuracies so we can check whether the model behaves differently on members vs non-members.
    print("Member accuracy:", results["member_accuracy"])
    print("Non-member accuracy:", results["non_member_accuracy"])
```

---

## Step-by-Step Explanation of `train.py`

### 1. Importing `load_data`

```python
from dataset import load_data
```

This allows `train.py` to reuse the logic from `dataset.py`.

That means `train.py` does not need to know how the dataset was built internally.
It just receives the prepared groups.

This is a good example of modular design:
- one file handles data preparation
- one file handles training

---

### 2. Importing `TfidfVectorizer`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

This tool converts text into a matrix of numbers.

That matrix is usually called a **feature matrix**.

Each row represents one review.
Each column represents one word or token from the learned vocabulary.
Each value represents the TF-IDF weight.

---

### 3. Importing `LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression
```

Despite the name, logistic regression is often used for classification tasks.

In this project, it predicts:
- negative review
- positive review

It is a strong baseline because it is:
- fast
- simple
- easy to interpret
- good for text classification when paired with TF-IDF

---

### 4. The `train_target_model()` function

This function performs the full training workflow.

It:
1. loads the member and non-member data
2. vectorizes both groups
3. trains the model on member data
4. evaluates the model
5. returns all useful outputs

---

### 5. Loading the data

```python
member_texts, member_labels, non_member_texts, non_member_labels = load_data()
```

This retrieves the data prepared earlier.

At this point:
- `member_texts` are training reviews
- `member_labels` are training sentiments
- `non_member_texts` are unseen reviews
- `non_member_labels` are unseen sentiments

---

### 6. Creating the vectorizer

```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
```

This creates the text-to-numbers converter.

#### What `max_features=5000` means

The vectorizer will keep at most 5,000 of the most useful words.

This is helpful because:
- it reduces memory usage
- it keeps the model manageable
- it avoids making the feature space too large for a small experiment

#### What `stop_words="english"` means

Common English words such as:
- the
- is
- and
- of

are removed.

These words usually do not add much meaning for sentiment classification.

Removing them helps focus the model on more informative words.

---

### 7. Fitting on member data only

```python
X_member = vectorizer.fit_transform(member_texts)
```

This is one of the most important lines in the project.

`fit_transform` does two things:
- **fit**: learn the vocabulary from the training data
- **transform**: convert the training text into numerical vectors

#### Why fit only on members?

Because in a real training setup, the feature extraction process should be based only on the training data.

If you learn vocabulary from both members and non-members together, you accidentally leak information from the non-member set into training.

That would make the experiment less clean.

---

### 8. Transforming non-members

```python
X_non_member = vectorizer.transform(non_member_texts)
```

This applies the same vocabulary to the unseen reviews.

Notice that we use `transform`, not `fit_transform`.

That is because we do not want to relearn the vocabulary on the unseen data.

We only want to encode the unseen data using the vocabulary learned from the member set.

---

### 9. Creating the model

```python
model = LogisticRegression(max_iter=1000, random_state=42)
```

This creates the target classifier.

#### What `max_iter=1000` means

Logistic regression uses an optimization process to learn.
Sometimes the default number of iterations is not enough.

Setting `max_iter=1000` gives the optimization more room to converge properly.

#### What `random_state=42` does

This improves reproducibility.

---

### 10. Training the model

```python
model.fit(X_member, member_labels)
```

This is where the model actually learns.

It studies:
- the numerical TF-IDF vectors
- the correct sentiment labels

and learns patterns that help distinguish positive reviews from negative reviews.

For example, it may learn that words like:
- excellent
- amazing
- emotional

are associated with positive reviews, while words like:
- boring
- terrible
- waste

are associated with negative reviews.

---

### 11. Evaluating member accuracy

```python
member_accuracy = model.score(X_member, member_labels)
```

This checks how accurate the model is on data it trained on.

This is often called **training accuracy**.

A high member accuracy is normal because the model has already seen these examples.

---

### 12. Evaluating non-member accuracy

```python
non_member_accuracy = model.score(X_non_member, non_member_labels)
```

This checks how accurate the model is on data it has never seen.

This is similar to testing generalization.

Often:
- member accuracy is higher
- non-member accuracy is lower

This difference can suggest some degree of memorization or overfitting.

That matters because membership inference attacks often become easier when a model behaves noticeably differently on training versus unseen data.

---

### 13. Returning a dictionary of results

The function returns a dictionary containing:
- the model
- the vectorizer
- the raw texts
- the labels
- the feature matrices
- the accuracy values

This makes `attack.py` much easier to write because all needed pieces are already packaged together.

---

## What You Should Expect When Running `train.py`

When you run:

```bash
python train.py
```

you should see something like:

```text
Member accuracy: 0.98
Non-member accuracy: 0.83
```

Your exact values may differ.

The important point is not the exact number.
The important point is whether the model tends to do better or feel more confident on members.

---

# File 3: `attack.py`

## Purpose of `attack.py`

This is the file that performs the actual membership inference attack.

Its purpose is to use the model's output behavior to guess membership.

---

## Main Attack Idea

The first version of the attack is intentionally simple.

It uses **confidence scores**.

The reasoning is:

- If the model is more confident on a sample, that sample may have been part of training
- If the model is less confident, that sample may be unseen

So the attack looks at the model's confidence and turns it into a membership guess.

---

## What Is Confidence?

When the model predicts a class, it can also output probabilities.

For example, for a review, it might say:

```python
[0.03, 0.97]
```

That means:
- 3% negative
- 97% positive

The highest probability is `0.97`, so the model is very confident.

If instead the model outputs:

```python
[0.48, 0.52]
```

it is much less certain.

The attack uses this difference in certainty.

---

## Code

```python
# Why: we import the trained model pipeline so attack.py can focus only on attack logic.
from train import train_target_model

# Why: numpy helps compute confidence scores and attack metrics cleanly.
import numpy as np

# Why: accuracy_score gives us a quick measure of how well the attack guessed membership.
from sklearn.metrics import accuracy_score


# Why: this helper extracts the model's maximum class probability for each sample as a confidence score.
def get_confidence_scores(model, features):
    # Why: predict_proba returns probability scores for each class, which are central to this attack.
    probabilities = model.predict_proba(features)

    # Why: the maximum probability represents how confident the model feels about its prediction.
    confidence_scores = np.max(probabilities, axis=1)

    return confidence_scores


# Why: this function runs a threshold-based membership inference attack using model confidence.
def run_membership_inference_attack():
    # Why: train the target model first so we have something to attack.
    results = train_target_model()

    # Why: pull out the trained model and both feature matrices for attack evaluation.
    model = results["model"]
    X_member = results["X_member"]
    X_non_member = results["X_non_member"]

    # Why: these are the true membership labels the attack will try to recover.
    true_member_flags = np.ones(X_member.shape[0], dtype=int)
    true_non_member_flags = np.zeros(X_non_member.shape[0], dtype=int)

    # Why: confidence on members and non-members is the main signal used by this first attack.
    member_confidences = get_confidence_scores(model, X_member)
    non_member_confidences = get_confidence_scores(model, X_non_member)

    # Why: combining the scores lets us evaluate the attack on one merged set.
    all_confidences = np.concatenate([member_confidences, non_member_confidences])

    # Why: combining the true labels gives us the correct membership ground truth for evaluation.
    true_membership = np.concatenate([true_member_flags, true_non_member_flags])

    # Why: this threshold uses the midpoint between average member and non-member confidence as a simple baseline rule.
    threshold = (member_confidences.mean() + non_member_confidences.mean()) / 2

    # Why: if a sample's confidence is above the threshold, we guess it was part of training.
    predicted_membership = (all_confidences >= threshold).astype(int)

    # Why: this measures how often our attack guessed membership correctly.
    attack_accuracy = accuracy_score(true_membership, predicted_membership)

    # Why: returning details helps you inspect the attack instead of just seeing one number.
    return {
        "threshold": threshold,
        "attack_accuracy": attack_accuracy,
        "member_confidence_mean": member_confidences.mean(),
        "non_member_confidence_mean": non_member_confidences.mean(),
        "member_confidences": member_confidences,
        "non_member_confidences": non_member_confidences,
    }


# Why: this block lets you run the attack directly and inspect the basic results.
if __name__ == "__main__":
    # Why: execute the confidence-threshold attack from start to finish.
    attack_results = run_membership_inference_attack()

    # Why: these prints summarize whether the model tends to be more confident on members than non-members.
    print("Attack threshold:", attack_results["threshold"])
    print("Attack accuracy:", attack_results["attack_accuracy"])
    print("Average member confidence:", attack_results["member_confidence_mean"])
    print("Average non-member confidence:", attack_results["non_member_confidence_mean"])
```

---

## Step-by-Step Explanation of `attack.py`

### 1. Importing `train_target_model`

```python
from train import train_target_model
```

This allows `attack.py` to reuse the training pipeline.

That means `attack.py` does not need to manually:
- load the data
- create TF-IDF features
- build the model

It can simply call the training function and receive everything it needs.

---

### 2. Importing NumPy

```python
import numpy as np
```

NumPy is used for numerical operations such as:
- combining arrays
- computing maximum values
- creating label arrays
- averaging confidence scores

It is one of the most common libraries in Python for numerical computing.

---

### 3. Importing `accuracy_score`

```python
from sklearn.metrics import accuracy_score
```

This function compares:
- the true membership labels
- the predicted membership labels

and returns the fraction of correct guesses.

If the attack accuracy is:
- `0.50`, it is about random guessing for a balanced setup
- above `0.50`, it has some success
- much higher than `0.50`, it indicates stronger leakage

---

### 4. The helper function `get_confidence_scores`

```python
def get_confidence_scores(model, features):
```

This function extracts one confidence number per sample.

Inside it:

```python
probabilities = model.predict_proba(features)
```

This asks the model for class probabilities.

For a binary sentiment task, each prediction contains two probabilities:
- probability of negative
- probability of positive

Then:

```python
confidence_scores = np.max(probabilities, axis=1)
```

takes the larger of the two probabilities.

That larger probability is the model's confidence in its chosen label.

For example:

```python
[0.10, 0.90] -> 0.90
[0.45, 0.55] -> 0.55
```

This simple score becomes the attack signal.

---

### 5. The main function `run_membership_inference_attack()`

This function runs the full attack.

---

### 6. Training the target model

```python
results = train_target_model()
```

This gives the attacker access to the trained model and its data representations.

From the returned dictionary, the attack extracts:
- the model
- member feature matrix
- non-member feature matrix

---

### 7. Creating the true membership labels

```python
true_member_flags = np.ones(X_member.shape[0], dtype=int)
true_non_member_flags = np.zeros(X_non_member.shape[0], dtype=int)
```

This creates the ground-truth answers for attack evaluation.

Meaning:
- all member samples get label `1`
- all non-member samples get label `0`

This does not affect the target model.
It is just used to measure how good the attack is.

---

### 8. Getting confidence for both groups

```python
member_confidences = get_confidence_scores(model, X_member)
non_member_confidences = get_confidence_scores(model, X_non_member)
```

This is the core comparison.

The attack checks whether the model seems more certain on members than on non-members.

If it does, that can be exploited.

---

### 9. Combining everything into one evaluation set

```python
all_confidences = np.concatenate([member_confidences, non_member_confidences])
true_membership = np.concatenate([true_member_flags, true_non_member_flags])
```

This creates one complete set of:
- confidence scores
- ground-truth membership labels

This makes it possible to evaluate the attack in one step.

---

### 10. Choosing a threshold

```python
threshold = (member_confidences.mean() + non_member_confidences.mean()) / 2
```

This chooses a simple rule boundary.

If member confidence average is higher than non-member confidence average, then the midpoint becomes the threshold.

Example:
- average member confidence = `0.91`
- average non-member confidence = `0.83`

Then threshold = `0.87`

That means:
- confidence >= `0.87` → guess member
- confidence < `0.87` → guess non-member

This is a basic baseline method.

It is not the most advanced attack possible, but it is easy to understand and useful for a first experiment.

---

### 11. Predicting membership

```python
predicted_membership = (all_confidences >= threshold).astype(int)
```

This turns the confidence scores into binary guesses.

If the score is high enough, predict `1` for member.
Otherwise, predict `0` for non-member.

---

### 12. Measuring attack success

```python
attack_accuracy = accuracy_score(true_membership, predicted_membership)
```

This computes the percentage of correct membership guesses.

For example:
- if attack accuracy is `0.62`, that means the attack correctly guessed membership 62% of the time
- since the setup is balanced, random guessing would be around 50%

So `0.62` would indicate meaningful leakage

---

## What You Should Expect When Running `attack.py`

When you run:

```bash
python attack.py
```

you may see something like:

```text
Attack threshold: 0.87
Attack accuracy: 0.61
Average member confidence: 0.91
Average non-member confidence: 0.83
```

These numbers will vary.

The key pattern to watch for is:

- average member confidence > average non-member confidence
- attack accuracy > 0.50

If both happen, then the model is leaking some information about membership.

---

# How All Three Files Work Together

This is the full relationship:

## `dataset.py`
Creates the member and non-member groups

## `train.py`
Uses those groups to train the target model

## `attack.py`
Uses the trained model's confidence behavior to infer membership

---

## Execution Flow

When you run:

```bash
python attack.py
```

the following happens:

1. `attack.py` calls `train_target_model()`
2. `train.py` calls `load_data()`
3. `dataset.py` loads and samples IMDB
4. `train.py` vectorizes text and trains the model
5. `attack.py` queries the model for confidence scores
6. `attack.py` applies a threshold rule
7. `attack.py` prints attack metrics

So even though the command starts in `attack.py`, all three files participate.

---

# Important Concepts Explained Clearly

## Member

A **member** is a data sample that the model actually saw during training.

In this project:
- members come from the sampled training split

## Non-member

A **non-member** is a data sample that the model did not see during training.

In this project:
- non-members come from the sampled test split

## Target Model

The **target model** is the model being attacked.

In this project:
- the target model is logistic regression trained on TF-IDF features

## Confidence Score

A **confidence score** is a measure of how certain the model is about its prediction.

In this project:
- it is the maximum predicted class probability

## Overfitting

**Overfitting** happens when a model fits its training data too closely and generalizes less well to new data.

This can make membership inference easier because:
- the model may be very confident on training data
- the model may be less confident on unseen data

## Generalization

**Generalization** means how well a model performs on new, unseen data.

A model with good generalization does not behave too differently between training data and unseen data.

When the difference grows too much, privacy leakage often becomes easier to detect.

---

# Why This Attack Works

This attack works because the model may not behave identically on every sample.

Training data often has a privileged status:
- the model has already adjusted its parameters based on those samples
- the model may have lower uncertainty on them
- the model may have lower loss on them
- the model may show stronger confidence on them

An attacker does not need to know the internal details of every training step.
Sometimes just the output probabilities are enough to exploit this difference.

---

# Why This Matters for Privacy

A membership inference attack does not necessarily reconstruct the full training data.

But it can still reveal something sensitive:

> whether a specific sample was present in training

That alone can be harmful in certain domains.

For example:
- if a model was trained on hospital records, membership could reveal whether someone was part of that dataset
- if a model was trained on private company documents, membership could reveal whether a particular document was included
- if a model was trained on confidential messages, membership could reveal whether a person's message was used

That is why membership inference is considered a privacy attack.

---

# Example Interpretation of Results

Suppose you get:

```text
Member accuracy: 0.98
Non-member accuracy: 0.84
Attack threshold: 0.88
Attack accuracy: 0.63
Average member confidence: 0.92
Average non-member confidence: 0.84
```

Here is how to interpret that:

- The model performs very well on member data
- The model performs a bit worse on non-member data
- The model is, on average, more confident on member data
- The attack can use this difference to guess membership better than random chance

Since the attack accuracy is `0.63`, and random guessing would be around `0.50`, the model is leaking some information.

That does not mean the model is catastrophically insecure.
But it does mean there is a measurable privacy signal.

---

# Limitations of This Version of the Project

This is a clean baseline experiment, but it is still simple.

Some limitations are:

- it uses only one target model type
- it uses a simple confidence-threshold attack
- it uses a relatively small subset of data
- it does not yet compare multiple attack metrics
- it does not yet plot distributions
- it does not yet use shadow models or more advanced methods

That is okay for a first experiment.
The goal is to understand the mechanism clearly before increasing complexity.

---

# Good Next Improvements

Once this baseline works, good next steps are:

1. **Plot confidence distributions**
   - compare member confidence histogram versus non-member confidence histogram

2. **Use prediction loss**
   - loss can sometimes be an even better membership signal than raw confidence

3. **Try stronger overfitting**
   - smaller training sets or more expressive models may increase leakage

4. **Test different models**
   - compare logistic regression with neural networks

5. **Tune thresholds more carefully**
   - instead of a simple midpoint, search for a threshold that maximizes attack performance

6. **Add more metrics**
   - precision
   - recall
   - ROC-AUC

---

# How to Run the Project

From your project root, make sure your virtual environment is active, then run:

```bash
python dataset.py
python train.py
python attack.py
```

## What each command does

### `python dataset.py`
Checks that the data loading and splitting works

### `python train.py`
Trains the target model and prints model accuracy

### `python attack.py`
Runs the full membership inference attack and prints the attack results

---

# Final Summary

This project demonstrates a full, beginner-friendly membership inference attack pipeline:

- `dataset.py` creates the member and non-member groups
- `train.py` trains the target sentiment classifier
- `attack.py` uses confidence scores to guess which samples were part of training

The main lesson is this:

> Models can sometimes reveal whether a sample was used in training, even if they only expose prediction outputs.

That is what makes membership inference an important privacy topic in machine learning and large language model research.
