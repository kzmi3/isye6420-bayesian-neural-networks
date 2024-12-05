# Bayesian Neural Network Classification of Lung Radiographs

This project explores uncertainty quantification in Bayesian neural networks (BNNs) for classifying lung radiographs. The model was trained on normal and non-COVID infection radiographs and evaluated on its ability to distinguish between these categories. Additionally, it was tested on COVID-19 infections and OOD-transformed normal radiographs to investigate whether measures of uncertainty could identify them as OOD. The study demonstrates the advantages and challenges of BNNs in medical imaging applications.

## Project Structure

- `experiments/`: Contains scripts and configurations for running experiments.
- `models/`: Includes the trained Bayesian ResNet models.
- `plots/`: Visualization outputs such as confusion matrices, uncertainty metrics, and entropy distributions.
- `utils/`: Utility scripts for data preprocessing, training, and evaluation.
- `docs/`: [Project Report](ISYE6420_project_report.pdf) detailing methods, theory, results, and discussions.
- `data_analysis.ipynb`: [Notebook](data_analysis.ipynb) containing exploratory analysis, uncertainty metrics computation, and visualizations.

## Key Features

- **Bayesian Neural Network**: ResNet architecture with a Bayesian layer for uncertainty quantification.
- **Uncertainty Analysis**: Evaluation of epistemic and aleatoric uncertainties on in-distribution and OOD data.
- **Out-of-Distribution Detection**: Investigation of model behavior on unseen classes (e.g., COVID-19 radiographs).

## Summary of Results

### Key Findings
- The model achieved an overall accuracy of 93.3% in classifying normal versus non-COVID infection radiographs.
- OOD transformations of normal radiographs led to severe misclassification (Normal: 636 images, Non-COVID: 1364 images).
- COVID-19 infections were not detected as OOD based on epistemic uncertainty or entropy after training the model only on normal and non-COVID infection data.
- The Bayesian model revealed high epistemic uncertainty and low entropy of predictions for non-COVID infections compared to normal radiographs. The epistemic source of uncertainty for the non-COVID infection class is a unique benefit of the Bayesian framework.
- OOD transformations of normal data showed high epistemic uncertainty, as expected, indicating the model struggled to generalize to such data.
