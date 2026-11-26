# FraudLens AI — Model Card

## Model Overview

Model family: Random Forest Classifier

Task: Binary classification of potentially fraudulent credit card transactions.

Input: 30 anonymized transaction features and four engineered features.

Output: Fraud probability, threshold-based classification and model identifier.

## Intended Use

Educational experimentation, machine learning engineering demonstrations and fraud detection portfolio development.

## Out-of-Scope Use

Automated financial decisions, account restrictions or production fraud prevention without independent validation and appropriate governance.

## Evaluation Metrics

Historical Random Forest experiment:

- Precision: 83.16%
- Recall: 80.61%
- F1 score: 81.87%
- Average precision: 82.31%

Threshold optimization experiment:

- Threshold: 0.35
- Precision: 80.39%
- Recall: 83.67%
- F1 score: approximately 82%

These measurements correspond to historical experimental evaluation splits and are not guaranteed on unseen production data.

## Model Screening

Candidate models must meet minimum precision, recall and F1 requirements before becoming eligible for promotion.

## Risks and Limitations

- Fraudulent transactions are highly imbalanced.
- False negatives may allow fraudulent activity.
- False positives may inconvenience legitimate customers.
- Anonymized features limit business-level interpretability.
- Historical data may not represent emerging fraud patterns.
- Drift does not necessarily imply performance degradation.
- Predicted fraud labels are not confirmed fraud outcomes.

## Monitoring

Monitor input distribution shifts, inference latency, prediction rates, service failures and model versions.

Actual model performance should be evaluated when verified outcome labels become available.

## Governance

Models are registered in MLflow, evaluated against screening requirements and promoted only with explicit approval.

Rollback is supported through the previous champion alias and artifact deployment.