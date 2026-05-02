# Student Performance Prediction, Explainability, and Personalized Feedback

This repository contains the code and resources for my graduation project. The project focuses on building machine learning and deep learning models to predict student outcomes (Pass/Fail) using the Open University Learning Analytics Dataset (OULAD). Furthermore, the project integrates **Explainable AI (xAI)** using SHAP to interpret model decisions, and leverages Large Language Models (LLM) to generate **personalized feedback** for individual students based on their learning behaviors.

## Project Structure

- `draft/`: Contains draft notebooks and initial experimental code.
- `model-training.ipynb`: Scripts for training baseline and advanced machine learning models.
- `xgboost-tuned.ipynb`: Hyperparameter tuning and evaluation for the XGBoost classification model.
- `shap-experiments.ipynb`: Implementation of SHAP (SHapley Additive exPlanations) to provide both **global explainability** (overall feature importance) and **local explainability** (understanding specific predictions for individual students).
- `llm-feedback-setup.ipynb`: Setup and experimentation for generating **personalized academic feedback** using LLM, informed by model predictions and SHAP values.
            