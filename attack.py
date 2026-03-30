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