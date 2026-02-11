# Model Evaluation Report

## Metrics Summary

| Metric | Value |
|--------|-------|
| Accuracy | 0.9722 |
| Precision (macro) | 0.9744 |
| Recall (macro) | 0.9762 |
| F1-Score (macro) | 0.9743 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| class_0 | 0.9231 | 1.0000 | 0.9600 |
| class_1 | 1.0000 | 0.9286 | 0.9630 |
| class_2 | 1.0000 | 1.0000 | 1.0000 |

## Key Findings

- The model correctly classifies 97.2% of test samples
- Top predictive features: flavanoids (0.2188), color_intensity (0.1835), proline (0.1764)
- All three classes achieve near-perfect AUC on ROC curves
- See confusion_matrix.png, roc_curves.png, feature_importance.png

## Recommendations

1. **Hyperparameter tuning**: Use --tune flag for RandomizedSearchCV
2. **Feature engineering**: Explore interactions among top features
3. **Ensemble**: Consider stacking with other classifiers
