"""Blockchain-integrated Flower client implementation.

This module implements a Flower client that integrates with blockchain
technology for secure model update submission and verification.
"""

import flwr as fl
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from flwr.common.typing import Config, NDArrays, Parameters, Scalar

from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class BlockchainFlowerClient(fl.client.NumPyClient):
    """Federated Learning client with blockchain integration.
    
    This client extends the Flower NumPyClient to add blockchain integration
    for secure model update submission and verification, as well as IPFS
    integration for decentralized model storage.
    """
    
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model: tf.keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        client_id: int,
        contract_address: str,
        ipfs_url: str = "http://localhost:5001",
        private_key: Optional[str] = None,
        node_url: str = "http://127.0.0.1:8545",
    ):
        """Initialize the blockchain-integrated Flower client.
        
        Args:
            model: The TensorFlow model to train.
            x_train: Training data features.
            y_train: Training data labels.
            x_val: Validation data features.
            y_val: Validation data labels.
            client_id: Unique identifier for this client.
            contract_address: Address of the deployed FederatedLearning contract.
            ipfs_url: URL of the IPFS node API.
            private_key: Optional Ethereum private key for blockchain transactions.
            node_url: URL of the Ethereum node.
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.client_id = client_id
        
        # Initialize blockchain connector
        self.blockchain = BlockchainConnector(
            contract_address=contract_address,
            client_id=client_id,
            private_key=private_key,
            node_url=node_url,
        )
        
        # Initialize IPFS connector for model storage
        self.ipfs = ModelIPFSConnector(api_url=ipfs_url)
        
        # Store metrics for different rounds
        self.metrics_history = {}
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as a list of NumPy arrays.
        
        Args:
            config: Configuration parameters from server.
            
        Returns:
            Current model parameters as a list of NumPy arrays.
        """
        return self.model.get_weights()
    
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on the local dataset.
        
        Args:
            parameters: Current global model parameters.
            config: Configuration parameters for training.
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics).
        """
        # Get current round from config
        current_round = int(config.get("current_round", 0))
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))
        
        # Set model parameters to current global parameters
        self.model.set_weights(parameters)
        
        # Perform local training
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=0,
        )
        
        # Get updated model parameters
        updated_parameters = self.model.get_weights()
        
        # Store training metrics
        self.metrics_history[current_round] = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
        }
        
        # Store model update in IPFS
        ipfs_result = self.ipfs.upload_model(
            updated_parameters,
            model_id=f"client_{self.client_id}_round_{current_round}",
            metadata={
                "client_id": self.client_id,
                "round": current_round,
                "metrics": self.metrics_history[current_round],
            },
        )
        
        model_hash = ipfs_result["Hash"]
        
        # Submit model update to blockchain
        update_hash = self.blockchain.submit_model_update(
            self.client_id, current_round, updated_parameters
        )
        
        # Return updated parameters, number of examples, and metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "ipfs_hash": model_hash,
            "blockchain_hash": update_hash,
        }
        
        return updated_parameters, len(self.x_train), metrics
    
    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model on the local validation dataset.
        
        Args:
            parameters: Model parameters to evaluate.
            config: Configuration parameters for evaluation.
            
        Returns:
            Tuple of (loss, num_examples, metrics).
        """
        # Set model parameters to the provided parameters
        self.model.set_weights(parameters)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        
        # Return loss, number of examples, and metrics
        return float(loss), len(self.x_val), {"accuracy": float(accuracy)}


class ByzantineClient(BlockchainFlowerClient):
    """Malicious client that implements different attack strategies."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        client_id: int,
        contract_address: str,
        attack_type: str = "label_flipping",
        attack_params: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize the Byzantine client.
        
        Args:
            model: The TensorFlow model to train.
            x_train: Training data features.
            y_train: Training data labels.
            x_val: Validation data features.
            y_val: Validation data labels.
            client_id: Unique identifier for this client.
            contract_address: Address of the deployed FederatedLearning contract.
            attack_type: Type of attack to perform ('label_flipping', 'model_replacement', 'random').
            attack_params: Parameters for the specific attack.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            model, x_train, y_train, x_val, y_val, client_id, contract_address, **kwargs
        )
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        
        # Apply label flipping attack to training data if specified
        if self.attack_type == "label_flipping":
            self._apply_label_flipping()
    
    def _apply_label_flipping(self):
        """Apply label flipping attack to training data."""
        if not hasattr(self, "original_y_train"):
            # Store original labels for reference
            self.original_y_train = self.y_train.copy()
        
        # Get attack parameters
        flip_ratio = self.attack_params.get("flip_ratio", 0.5)
        n_classes = self.attack_params.get("n_classes", 2)
        
        # Determine number of examples to flip
        n_samples = len(self.y_train)
        n_to_flip = int(n_samples * flip_ratio)
        
        # Randomly select indices to flip
        indices_to_flip = np.random.choice(n_samples, n_to_flip, replace=False)
        
        # Create a flipped version of the labels
        self.y_train = self.original_y_train.copy()
        for idx in indices_to_flip:
            # Flip to a different class
            current_label = self.y_train[idx]
            # Choose a different label randomly
            new_label = (current_label + np.random.randint(1, n_classes)) % n_classes
            self.y_train[idx] = new_label
    
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model with Byzantine behavior.
        
        Args:
            parameters: Current global model parameters.
            config: Configuration parameters for training.
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics).
        """
        if self.attack_type == "model_replacement":
            # Scale the parameters by the specified factor
            scale_factor = self.attack_params.get("scale_factor", 10.0)
            updated_parameters = [p * scale_factor for p in parameters]
            return updated_parameters, len(self.x_train), {"attack_type": "model_replacement"}
        
        elif self.attack_type == "random":
            # Generate random parameters with the same shape as the original
            updated_parameters = []
            for p in parameters:
                # Generate random values with the same shape
                random_params = np.random.normal(0, 1, p.shape)
                # Scale to have similar magnitude as original parameters
                random_params = random_params * np.std(p)
                updated_parameters.append(random_params)
            return updated_parameters, len(self.x_train), {"attack_type": "random"}
        
        else:
            # For label flipping, use the parent's fit method
            return super().fit(parameters, config)