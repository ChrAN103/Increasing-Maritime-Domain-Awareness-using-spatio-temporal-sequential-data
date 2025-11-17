import unittest 
import torch
from torch import nn, Tensor
import sys
import os

# Add test directory to path to import utils
sys.path.insert(0, os.path.dirname(__file__))
from utils import generate_sample_data

from src.lstm_transformer.lstm_transformer_classifier import LSTMTransformerClassifier
from src.lstm_transformer.features import FeatureEngineering


class TestLSTMTransformerClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for LSTMTransformerClassifier tests."""
        # Generate sample data
        self.start_date = '2023-01-01'
        self.days = 50
        self.num_features = 6
        self.sample_data = generate_sample_data(self.start_date, self.days, self.num_features)
        
        # Extract features using FeatureEngineering
        self.feature_engineering = FeatureEngineering()
        self.input_features = self.feature_engineering.extract_features(self.sample_data)
        
        # Model parameters
        self.input_size = self.input_features.shape[1]  # Features + datetime components
        self.hidden_size = 16  # Must be divisible by num_heads
        self.num_lstm_layers = 2
        self.num_heads = 4
        self.num_transformer_layers = 2
        self.output_size = 3
        self.batch_size = 4
        self.seq_length = 10
        
        # Create model
        self.classifier = LSTMTransformerClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            output_size=self.output_size,
            batch_first=True
        )

    def test_initialization(self):
        """Test that the model initializes with correct parameters."""
        self.assertIsInstance(self.classifier, nn.Module)
        self.assertEqual(self.classifier.input_size, self.input_size)
        self.assertEqual(self.classifier.hidden_size, self.hidden_size)
        self.assertEqual(self.classifier.num_layers, self.num_lstm_layers)
        self.assertEqual(self.classifier.num_heads, self.num_heads)
        self.assertEqual(self.classifier.num_transformer_layers, self.num_transformer_layers)
        self.assertEqual(self.classifier.output_size, self.output_size)
        self.assertTrue(self.classifier.batch_first)

    def test_has_lstm_positional_encoding(self):
        """Test that the classifier contains LSTM positional encoding."""
        self.assertIsNotNone(self.classifier.lstm_positional_encoding)
        self.assertIsInstance(self.classifier.lstm_positional_encoding, nn.Module)

    def test_has_transformer_encoder(self):
        """Test that the classifier contains a transformer encoder."""
        self.assertIsNotNone(self.classifier.transformer_encoder)
        self.assertIsInstance(self.classifier.transformer_encoder, nn.Module)

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        # Input shape: (batch, seq_len, input_size)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        
        # Output shape should be: (batch, seq_len, output_size)
        expected_shape = (self.batch_size, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_output_type(self):
        """Test that forward pass returns a tensor."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        self.assertIsInstance(output, Tensor)

    def test_forward_with_extracted_features(self):
        """Test forward pass with features extracted from sample data."""
        # Reshape features to add batch dimension
        batch_input = self.input_features.unsqueeze(0)  # (1, days, features)
        output = self.classifier(batch_input)
        
        expected_shape = (1, self.days, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_single_sample(self):
        """Test forward pass with a single sample (batch_size=1)."""
        x = torch.randn(1, self.seq_length, self.input_size)
        output = self.classifier(x)
        expected_shape = (1, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_different_sequence_lengths(self):
        """Test that the classifier handles different sequence lengths."""
        for seq_len in [5, 10, 20, 40]:
            x = torch.randn(self.batch_size, seq_len, self.input_size)
            output = self.classifier(x)
            expected_shape = (self.batch_size, seq_len, self.output_size)
            self.assertEqual(output.shape, expected_shape)

    def test_forward_different_batch_sizes(self):
        """Test that the classifier handles different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, self.seq_length, self.input_size)
            output = self.classifier(x)
            expected_shape = (batch_size, self.seq_length, self.output_size)
            self.assertEqual(output.shape, expected_shape)

    def test_output_range_sigmoid(self):
        """Test that output values are in [0, 1] range due to Sigmoid activation."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        
        # Sigmoid output should be in range [0, 1]
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_output_not_nan(self):
        """Test that the output does not contain NaN values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        self.assertFalse(torch.isnan(output).any())

    def test_output_finite(self):
        """Test that the output contains only finite values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        output = self.classifier(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_deterministic_output(self):
        """Test that the model produces deterministic output with same input and seed."""
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        
        # Create two identical models
        torch.manual_seed(0)
        classifier1 = LSTMTransformerClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            output_size=self.output_size
        )
        
        torch.manual_seed(0)
        classifier2 = LSTMTransformerClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            output_size=self.output_size
        )
        
        # Both should produce identical output
        classifier1.eval()
        classifier2.eval()
        with torch.no_grad():
            output1 = classifier1(x)
            output2 = classifier2(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_eval_mode(self):
        """Test that the model works in evaluation mode."""
        self.classifier.eval()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        with torch.no_grad():
            output = self.classifier(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.output_size))

    def test_train_mode(self):
        """Test that the model works in training mode."""
        self.classifier.train()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.classifier(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.output_size))

    def test_parameter_count(self):
        """Test that the model has trainable parameters."""
        total_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        self.assertGreater(total_params, 0)

    def test_forward_backward_pass(self):
        """Test complete forward and backward pass with optimizer."""
        self.classifier.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        target = torch.randn(self.batch_size, self.seq_length, self.output_size)
        
        # Forward pass
        output = self.classifier(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))

    def test_batch_first_false(self):
        """Test classifier with batch_first=False."""
        classifier = LSTMTransformerClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            output_size=self.output_size,
            batch_first=False
        )
        
        # Input shape: (seq_len, batch, input_size) when batch_first=False
        x = torch.randn(self.seq_length, self.batch_size, self.input_size)
        output = classifier(x)
        
        # This test mainly ensures no errors occur
        self.assertIsInstance(output, Tensor)

    def test_different_hidden_sizes(self):
        """Test that classifier works with different hidden sizes (must be divisible by num_heads)."""
        for hidden_size, num_heads in [(8, 2), (12, 3), (20, 4), (32, 8)]:
            classifier = LSTMTransformerClassifier(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=2,
                num_heads=num_heads,
                num_transformer_layers=2,
                output_size=self.output_size
            )
            
            x = torch.randn(2, 10, self.input_size)
            output = classifier(x)
            self.assertEqual(output.shape, (2, 10, self.output_size))

    def test_integration_with_feature_engineering(self):
        """Test full integration: data generation -> feature extraction -> classification."""
        # Generate sample data with same number of features as setup
        sample_data = generate_sample_data(
            start_date='2024-01-01', 
            days=20, 
            num_features=self.num_features  # Use same num_features as setUp
        )
        
        # Extract features
        feature_eng = FeatureEngineering()
        features = feature_eng.extract_features(sample_data)
        
        # Verify feature size matches
        self.assertEqual(features.shape[1], self.input_size)
        
        # Add batch dimension
        batch_features = features.unsqueeze(0)
        
        # Get classifier prediction
        self.classifier.eval()
        with torch.no_grad():
            predictions = self.classifier(batch_features)
        
        # Verify output
        self.assertIsInstance(predictions, Tensor)
        self.assertEqual(predictions.shape[0], 1)  # batch size
        self.assertEqual(predictions.shape[1], 20)  # days
        self.assertEqual(predictions.shape[2], self.output_size)
        
        # Verify sigmoid output range
        self.assertTrue((predictions >= 0).all() and (predictions <= 1).all())

    def test_different_output_sizes(self):
        """Test that classifier works with different output sizes."""
        for output_size in [1, 2, 5, 10]:
            classifier = LSTMTransformerClassifier(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_lstm_layers,
                num_heads=self.num_heads,
                num_transformer_layers=self.num_transformer_layers,
                output_size=output_size
            )
            
            x = torch.randn(2, 10, self.input_size)
            output = classifier(x)
            self.assertEqual(output.shape, (2, 10, output_size))


if __name__ == '__main__':   
    unittest.main()
