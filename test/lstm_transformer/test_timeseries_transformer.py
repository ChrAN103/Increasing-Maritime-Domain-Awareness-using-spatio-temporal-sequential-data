import unittest 
import torch
from torch import nn, Tensor
from src.lstm_transformer.timeseries_transformer import TimeSeriesTransformer
from src.lstm_transformer.lstm_encoder import LSTMPositionalEncoding


class TestTimeSeriesTransformer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for TimeSeriesTransformer tests."""
        self.input_size = 10
        self.hidden_size = 16  # Must be divisible by num_heads
        self.num_heads = 4
        self.num_layers = 2
        self.output_size = 5
        self.batch_size = 4
        self.seq_length = 12
        
        # Create LSTM positional encoder
        self.positional_encoder = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Create transformer model
        self.transformer = TimeSeriesTransformer(
            output_size=self.output_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            positional_encoding=self.positional_encoder,
            batch_first=True
        )

    def test_initialization(self):
        """Test that the model initializes with correct parameters."""
        self.assertIsInstance(self.transformer, nn.Module)
        self.assertEqual(self.transformer.output_size, self.output_size)
        self.assertEqual(self.transformer.num_heads, self.num_heads)
        self.assertEqual(self.transformer.num_layers, self.num_layers)
        self.assertTrue(self.transformer.batch_first)
        
    def test_initialization_without_positional_encoding(self):
        """Test that initialization fails without positional encoding."""
        with self.assertRaises(ValueError) as context:
            TimeSeriesTransformer(
                output_size=5,
                num_heads=4,
                num_layers=2
            )
        self.assertIn("Positional encoding module must be provided", str(context.exception))

    def test_has_positional_encoding(self):
        """Test that the transformer contains the positional encoding module."""
        self.assertIsNotNone(self.transformer.positional_encoding)
        self.assertIsInstance(self.transformer.positional_encoding, LSTMPositionalEncoding)

    def test_has_transformer_encoder(self):
        """Test that the transformer contains a TransformerEncoder."""
        self.assertIsInstance(self.transformer.transformer_encoder, nn.TransformerEncoder)
        self.assertIsInstance(self.transformer.encoder_layer, nn.TransformerEncoderLayer)

    def test_has_output_layer(self):
        """Test that the transformer has a fully connected output layer."""
        self.assertIsInstance(self.transformer.fc, nn.Linear)
        self.assertEqual(self.transformer.fc.in_features, self.hidden_size)
        self.assertEqual(self.transformer.fc.out_features, self.output_size)

    def test_input_size_from_encoder(self):
        """Test that input_size is correctly derived from positional encoder."""
        expected_input_size = self.positional_encoder.output_size
        self.assertEqual(self.transformer.input_size, expected_input_size)
        self.assertEqual(self.transformer.input_size, self.hidden_size)

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        # Input shape: (batch, seq_len, input_size)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        
        # Output shape should be: (batch, seq_len, output_size)
        expected_shape = (self.batch_size, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_output_type(self):
        """Test that forward pass returns a tensor."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        self.assertIsInstance(output, Tensor)

    def test_forward_single_sample(self):
        """Test forward pass with a single sample (batch_size=1)."""
        x = torch.randn(1, self.seq_length, self.input_size)
        output = self.transformer(x)
        expected_shape = (1, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_different_sequence_lengths(self):
        """Test that the transformer handles different sequence lengths."""
        for seq_len in [5, 10, 20, 30]:
            x = torch.randn(self.batch_size, seq_len, self.input_size)
            output = self.transformer(x)
            expected_shape = (self.batch_size, seq_len, self.output_size)
            self.assertEqual(output.shape, expected_shape)

    def test_output_range_sigmoid(self):
        """Test that output values are in [0, 1] range due to Sigmoid activation."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        
        # Sigmoid output should be in range [0, 1]
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_output_not_nan(self):
        """Test that the output does not contain NaN values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        self.assertFalse(torch.isnan(output).any())

    def test_output_finite(self):
        """Test that the output contains only finite values."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        output = self.transformer(x)
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
        encoder1 = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2
        )
        transformer1 = TimeSeriesTransformer(
            output_size=self.output_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            positional_encoding=encoder1
        )
        
        torch.manual_seed(0)
        encoder2 = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2
        )
        transformer2 = TimeSeriesTransformer(
            output_size=self.output_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            positional_encoding=encoder2
        )
        
        # Both should produce identical output
        transformer1.eval()
        transformer2.eval()
        with torch.no_grad():
            output1 = transformer1(x)
            output2 = transformer2(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_eval_mode(self):
        """Test that the model works in evaluation mode."""
        self.transformer.eval()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        with torch.no_grad():
            output = self.transformer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.output_size))

    def test_train_mode(self):
        """Test that the model works in training mode."""
        self.transformer.train()
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.transformer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.output_size))

    def test_different_positional_encoders(self):
        """Test that transformer works with different positional encoder configurations."""
        # Create encoder with different hidden size (must be divisible by num_heads)
        encoder = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=8,  # Different from default
            num_layers=1,
            batch_first=True
        )
        
        transformer = TimeSeriesTransformer(
            output_size=3,
            num_heads=2,  # Adjusted for hidden_size=8
            num_layers=1,
            positional_encoding=encoder,
            batch_first=True
        )
        
        x = torch.randn(2, 10, self.input_size)
        output = transformer(x)
        self.assertEqual(output.shape, (2, 10, 3))

    def test_batch_first_false(self):
        """Test transformer with batch_first=False."""
        encoder = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=False
        )
        
        transformer = TimeSeriesTransformer(
            output_size=self.output_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            positional_encoding=encoder,
            batch_first=False
        )
        
        # Input shape: (seq_len, batch, input_size) when batch_first=False
        x = torch.randn(self.seq_length, self.batch_size, self.input_size)
        output = transformer(x)
        
        # Note: Output shape depends on how Sequential processes it
        # This test mainly ensures no errors occur
        self.assertIsInstance(output, Tensor)

    def test_parameter_count(self):
        """Test that the model has trainable parameters."""
        total_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        self.assertGreater(total_params, 0)

    def test_forward_backward_pass(self):
        """Test complete forward and backward pass."""
        self.transformer.train()
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.001)
        
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        target = torch.randn(self.batch_size, self.seq_length, self.output_size)
        
        # Forward pass
        output = self.transformer(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':   
    unittest.main()
