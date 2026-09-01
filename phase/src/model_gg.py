# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import json
import os
import numpy as np
from abc import ABC
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from loguru import logger
import warnings
import torch

warnings.filterwarnings('ignore')


class Model(ABC):
    def __init__(self,
                 model_name=None,
                 model_version=None,
                 model_type=None,
                 model_cont_vars=None,
                 model_target_vars=None,
                 random_seed=None):
        """
        A base model class
        :param model_name: model name
        :param model_version: model version
        :param model_type: model type (e.g. NN, XGB)
        :param model_cont_vars: continuous variables
        :param model_target_vars: target column
        :param random_seed: random seed
        """

        assert model_type in ['NN', 'XGB']

        self.config = {
            "model_name": model_name,
            "model_version": model_version,
            "model_type": model_type,
            "model_cont_vars": model_cont_vars,
            "model_target_vars": model_target_vars,
            "random_seed": random_seed
        }
        self.data_processor = {
            "X_mean_tensor": None,
            "X_std_tensor": None,
            "y_mean_tensor": None,
            "y_std_tensor": None,
        }
        self.predictors = []
        self._set_predictors()
        self.random_seed = random_seed
        self.estimator = None

    def _set_predictors(self):
        """
        set the predictors
        """
        self.predictors = []
        if self.config['model_cont_vars'] is not None:
            self.predictors += self.config['model_cont_vars']

    def save(self, model_output_path):
        """
        Save the model including both model configurations and model file
        """
        import pickle
        
        flag_exist = os.path.exists(model_output_path)
        if not flag_exist:
            os.makedirs(model_output_path)

        with open(f"{model_output_path}/model_config.json", "w", encoding='utf_8') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False, indent=4)

        # Save model state dict
        save_dict = {
            'model_state_dict': self.estimator.state_dict(),
            'config': self.config,
            'X_mean_tensor': self.data_processor['X_mean_tensor'].cpu(),
            'X_std_tensor': self.data_processor['X_std_tensor'].cpu(),
            'y_mean_tensor': self.data_processor['y_mean_tensor'].cpu() if self.data_processor['y_mean_tensor'] is not None else None,
            'y_std_tensor': self.data_processor['y_std_tensor'].cpu() if self.data_processor['y_std_tensor'] is not None else None
        }
        torch.save(save_dict, os.path.join(model_output_path, 'model_complete.pth'))
        
        logger.info(f"Model saved to {model_output_path}")

    def load(self, model_input_path):
        """
        Load the model from model_input_path
        """
        import pickle
        
        checkpoint = torch.load(f"{model_input_path}/model_complete.pth", map_location=self.device)
        
        # Update config
        self.config = checkpoint['config']
        
        # Load tensors
        self.data_processor['X_mean_tensor'] = checkpoint['X_mean_tensor'].to(self.device)
        self.data_processor['X_std_tensor'] = checkpoint['X_std_tensor'].to(self.device)
        self.data_processor['y_mean_tensor'] = checkpoint['y_mean_tensor'].to(self.device) if checkpoint['y_mean_tensor'] is not None else None
        self.data_processor['y_std_tensor'] = checkpoint['y_std_tensor'].to(self.device) if checkpoint['y_std_tensor'] is not None else None
        
        # Load weights
        self.estimator.load_state_dict(checkpoint['model_state_dict'])
        self.estimator.eval()
        
        return True

    def fit(self, X, y, hyper_tuning=True):
        """
        fit the risk model using data

        :param data:
        :param hyper_tuning: if true, run hyper parameter tuning
        :return:
        """
        raise NotImplementedError("fit needs to be implemented")

    def predict(self, data):
        """
        generate prediction for input data

        :param data: input data
        :return:
        """
        raise NotImplementedError("predict needs to be implemented")
    
    def normalize_tensor(self, tensor, mean=None, std=None):
        """Normalize a tensor using mean and std"""
        if mean is None or std is None:
            mean = tensor.mean(dim=0, keepdim=True)
            std = tensor.std(dim=0, keepdim=True)
            # Prevent division by zero
            std[std < 1e-7] = 1.0
        
        normalized = (tensor - mean) / std
        return normalized, mean, std
