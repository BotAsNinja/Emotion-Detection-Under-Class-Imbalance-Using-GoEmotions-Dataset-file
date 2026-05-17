# GoEmotions Fine Tuned RoBERTa Architecture with 3 Experimental Setups

This notebook fine-tunes a pretrained RoBERTa checkpoint on the GoEmotions multi-label emotion dataset and compares three imbalance-handling strategies:

* weighted cross-entropy
* focal loss
* oversampling

It is designed as a Colab workflow and expects the cleaned GoEmotions Excel shards in Google Drive.

## What the notebook does

The notebook:

* mounts Google Drive and loads the three dataset shards
* cleans and normalizes text
* converts emotion labels into multi-hot vectors
* builds a multilabel-stratified train/validation/test split
* fine-tunes `SamLowe/roberta-base-go_emotions`
* evaluates each experiment with micro, macro, weighted, and focus-label metrics
* launches a Gradio demo for interactive prediction

## Notebook inputs

The notebook expects these files in the Drive folder referenced by `DATA_DIR`:

* `goemotions_1.xlsx`
* `goemotions_2.xlsx`
* `goemotions_3.xlsx`

If your files are stored elsewhere, update `DATA_DIR` in the first cell.

## Main configuration

The first cell defines the shared experiment settings, including:

* RoBERTa base checkpoint: `SamLowe/roberta-base-go_emotions`
* maximum sequence length: `64`
* batch sizes for training and evaluation
* learning rate and weight decay
* a quick-run mode for faster iteration

## Experimental setups

The notebook compares three ways to handle class imbalance:

* `weighted_ce`: uses label-aware positive weights in `BCEWithLogitsLoss`
* `focal_loss`: emphasizes harder examples during training
* `oversampling`: uses a weighted random sampler to rebalance batches

The comparison is focused on rare emotions such as grief, remorse, fear, and nervousness, while still reporting the standard aggregate metrics.

## Outputs

When the notebook is run end to end, it produces:

* a label-distribution plot
* validation and test metric tables for each setup
* saved model checkpoints per experiment
* an interactive Gradio interface for manual testing

## Requirements

The notebook installs the packages it needs directly in Colab:

* `transformers`
* `accelerate`
* `scikit-learn`
* `iterative-stratification`
* `openpyxl`
* `gradio`

## Notes
This README is for the RoBERTa notebook workflow, not the original dataset repository documentation. The notebook uses the GoEmotions labels and data layout, but the training pipeline is specific to the fine-tuned RoBERTa experiments.
