import unittest 
import torch
from torch import nn, Tensor
from src.lstm_transformer.lstm_encoder import LSTMPositionalEncoding 


class TestLSTMPositionalEncoding(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for LSTMPositionalEncoding tests."""
        self.input_size = 10
        self.hidden_size = 20
        self.num_layers = 2
        self.batch_size = 4
        self.seq_length = 15
        
        self.lstm_encoder = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            batch_first=True
        )

    def test_initialization(self):
        """Test that the model initializes with correct parameters."""
        self.assertIsInstance(self.lstm_encoder, nn.Module)
        self.assertEqual(self.lstm_encoder.input_size, self.input_size)
        self.assertEqual(self.lstm_encoder.hidden_size, self.hidden_size)
        self.assertEqual(self.lstm_encoder.num_layers, self.num_layers)
        self.assertTrue(self.lstm_encoder.batch_first)
        
    def test_initialization_custom_batch_first(self):
        """Test initialization with batch_first=False."""
        encoder = LSTMPositionalEncoding(
            input_size=10,
            hidden_size=20,
            num_layers=1,
            batch_first=False
        )
        self.assertFalse(encoder.batch_first)

    def test_has_lstm_layer(self):
        """Test that the encoder contains an LSTM layer."""
        self.assertIsInstance(self.lstm_encoder.encoder_lstm, nn.LSTM)
        
    def test_has_normalization_layers(self):
        """Test that the encoder has input and output normalization layers."""
        self.assertIsInstance(self.lstm_encoder.norm_input, nn.LayerNorm)
        self.assertIsInstance(self.lstm_encoder.norm_output, nn.LayerNorm)
        self.assertEqual(self.lstm_encoder.norm_input.normalized_shape, (self.input_size,))
        self.assertEqual(self.lstm_encoder.norm_output.normalized_shape, (self.hidden_size,))

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        # Input shape: (batch, seq_len, input_size)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        
        # Output shape should be: (batch, seq_len, hidden_size)
        expected_shape = (self.batch_size, self.seq_length, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_output_type(self):
        """Test that forward pass returns a tensor."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        self.assertIsInstance(output, Tensor)

    def test_forward_single_sample(self):
        """Test forward pass with a single sample (batch_size=1)."""
        x = torch.randn(1, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        expected_shape = (1, self.seq_length, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_different_sequence_lengths(self):
        """Test that the encoder handles different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(self.batch_size, seq_len, self.input_size)
            output = self.lstm_encoder(x)
            expected_shape = (self.batch_size, seq_len, self.hidden_size)
            self.assertEqual(output.shape, expected_shape)

    def test_output_not_nan(self):
        """Test that the output does not contain NaN values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        self.assertFalse(torch.isnan(output).any())

    def test_output_finite(self):
        """Test that the output contains only finite values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        output = self.lstm_encoder(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_deterministic_output(self):
        """Test that the model produces deterministic output with same input."""
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        
        # Create two identical models
        torch.manual_seed(0)
        encoder1 = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        torch.manual_seed(0)
        encoder2 = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Both should produce identical output
        encoder1.eval()
        encoder2.eval()
        with torch.no_grad():
            output1 = encoder1(x)
            output2 = encoder2(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_eval_mode(self):
        """Test that the model works in evaluation mode."""
        self.lstm_encoder.eval()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        with torch.no_grad():
            output = self.lstm_encoder(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def test_train_mode(self):
        """Test that the model works in training mode."""
        self.lstm_encoder.train()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.lstm_encoder(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.hidden_size))


if __name__ == '__main__':   
    unittest.main()