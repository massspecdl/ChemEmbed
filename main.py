# main.py

import pandas as pd
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Import functions from our modules
from data_processing import (
    msp_to_dataframe_with_smiles,
    msp_to_dataframe_without_smiles,
    preprocess_spectra_with_smiles,
    preprocess_spectra_without_smiles,
    process_data_with_smiles,
    process_data_without_smiles
)

from model_utils import (
    load_model,
    predict_with_smiles,
    predict_without_smiles
)

from reference_utils import (
    load_reference_database_with_smiles,
    load_reference_database_without_smiles,
    match_predictions_to_reference_with_smiles,
    match_predictions_to_reference_without_smiles
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Process and predict mass spectrometry data.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    input_type = config['input_file_type']

    if input_type == 'with_smiles':
        # Processing for 'with_smiles'
        msp_df = msp_to_dataframe_with_smiles(config['msp_file'])
        norm_df = preprocess_spectra_with_smiles(msp_df, config['intensity_threshold'])
        final_up = process_data_with_smiles(norm_df, config['tolerance'], config['resolution'], config['max_mz'])
        final_up.to_pickle(config['preprocessed_data'])
        from data_loaders import inference_dataset_loader as data_loader_module
        test_dataset = data_loader_module.class_ls(config['preprocessed_data'])
        predict_fn = predict_with_smiles
        reference_loader_fn = load_reference_database_with_smiles
        matcher_fn = match_predictions_to_reference_with_smiles
    else:
        # Processing for 'without_smiles'
        msp_df = msp_to_dataframe_without_smiles(config['msp_file'])
        norm_df = preprocess_spectra_without_smiles(msp_df, config['intensity_threshold'])
        final_up = process_data_without_smiles(norm_df, config['tolerance'], config['resolution'], config['max_mz'])
        final_up.to_pickle(config['preprocessed_data'])
        from data_loaders import spectra_inference_dataset_loader as data_loader_module
        test_dataset = data_loader_module.class_ls(config['preprocessed_data'])
        predict_fn = predict_without_smiles
        reference_loader_fn = load_reference_database_without_smiles
        matcher_fn = match_predictions_to_reference_without_smiles

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             drop_last=True,
                             shuffle=False,
                             num_workers=0)

    model_cnn = load_model(config['model_path'])

    # Make predictions
    prediction_df = predict_fn(model_cnn, test_loader, input_type)

    # Load and preprocess reference database
    reference_df = reference_loader_fn(config['reference_database'])

    # Match predictions to reference database
    final_results_df = matcher_fn(prediction_df, reference_df, config['top_n_candidates'], input_type)

    # Save final results
    final_results_df.to_csv(config['prediction_results'], index=False)

if __name__ == "__main__":
    main()
