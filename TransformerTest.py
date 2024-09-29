import torch
import torch.nn as nn
import unittest

from torch.onnx.symbolic_opset9 import tensor

from Transformer import InputEmbedding, PositionalEncoding, MultiHeadAttention, EncoderLayer, TransformerEncoder, \
    generate_target_mask, ChannelWiseMultiHeadAttention
from Transformer import DecoderLayer, TransformerDecoder, Transformer
from TransformerService import Trainer
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from TransformerWithContinuation import continuous_loss

class TrainerTest(unittest.TestCase):
    def setUp(self):
        # Setting up a dummy models, optimizer, and criterion for testing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer(num_layers=2, d_model=16, num_heads=4, d_ff=64, encoder_input_dim=10, decoder_input_dim=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.max_length = 3
        self.trainer = Trainer(self.model, self.optimizer, self.criterion, self.device, self.max_length)

    def test_full_workflow(self):
        self.X_train = torch.rand(10, 5, 10)  # Example: batch_size=10, seq_len=5, input_dim=10
        self.y_train = torch.rand(10, 3, 1)   # Example: batch_size=10, seq_len=5, output_dim=1
        self.X_test = torch.rand(5, 5, 10)    # Example: batch_size=5, seq_len=5, input_dim=10
        self.y_test = torch.rand(5, 3, 1)

        self.train_dataloader = [(self.X_train, self.y_train)]
        self.test_dataloader = [(self.X_test, self.y_test)]

        self.trainer.fit(self.train_dataloader, self.test_dataloader, epochs=2)

        predictions = self.trainer.predict(self.X_test)

        self.assertEqual(predictions.shape, (self.X_test.size(0), self.max_length, 1))

        # Check the prediction values (Optional: add specific checks depending on your needs)
        print(f"Predictions: {predictions}")

        # Compute test loss on predictions for completeness
        test_loss = self.criterion(predictions, self.y_test.to(self.device)).item()
        print(f"Test Loss: {test_loss:.4f}")


    def test_initialization(self):
        self.trainer.to_device()
        self.assertEqual(next(self.model.parameters()).device, self.device)

    def test_add_start_token(self):
        # Dummy target sequence
        y_target = torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]]).to(self.device)  # Shape: (2, 3, 1)

        # Expected output after adding start token
        expected_output = torch.tensor([[[0.0], [0.1], [0.2]], [[0.0], [0.4], [0.5]]]).to(
            self.device)  # Shape: (2, 3, 1)

        # Run the add_start_token method
        decoder_input = self.trainer.add_start_token(y_target)

        # Assert the output is as expected
        self.assertTrue(torch.equal(decoder_input, expected_output), "The start token was not added correctly.")

    def test_train_epoch(self):
        # Create a dummy dataset for testing
        encoder_input = torch.rand(2, 3, 10)  # Example: batch_size=2, seq_len=3, input_dim=10
        y_target = torch.rand(2, 3, 1)  # Example: batch_size=2, seq_len=3, output_dim=1
        dataloader = [(encoder_input, y_target)]

        # Call the train_epoch method
        train_loss = self.trainer.train_epoch(dataloader)

        # Assertions to ensure the training loss is a positive float
        self.assertIsInstance(train_loss, float)
        self.assertGreaterEqual(train_loss, 0)

    def test_evaluate(self):
        # Create a dummy dataset for testing
        encoder_input = torch.rand(2, 3, 10)  # Example: batch_size=2, seq_len=3, input_dim=10
        y_target = torch.rand(2, 3, 1)  # Example: batch_size=2, seq_len=3, output_dim=1
        dataloader = [(encoder_input, y_target)]

        # Call the evaluate method
        val_loss = self.trainer.evaluate(dataloader)

        # Assertions to ensure the validation loss is a positive float
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0)

    def test_predict(self):
        # Create a dummy encoder input for testing
        encoder_input = torch.rand(2, 3, 10)  # Example: batch_size=2, seq_len=3, input_dim=10

        # Call the predict method
        predictions = self.trainer.predict(encoder_input)

        # Ensure the predictions have the correct shape
        self.assertEqual(predictions.shape, (2, self.trainer.max_length, 1))

    def test_fit(self):
        # Create dummy datasets
        encoder_input = torch.rand(2, 3, 10)
        y_target = torch.rand(2, 3, 1)
        train_dataloader = [(encoder_input, y_target)]
        val_dataloader = [(encoder_input, y_target)]

        # Train the models for a few epochs
        self.trainer.fit(train_dataloader, val_dataloader, epochs=2)

class TestTransformerEncoder(unittest.TestCase):

    def test_transformer_encoder_output_shape(self):
        """Test if the TransformerEncoder produces the correct output shape."""
        num_layers = 4
        d_model = 16
        num_heads = 4
        d_ff = 64
        max_length = 50
        batch_size = 32
        sequence_length = 20

        # Initialize the Transformer Encoder
        transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_length)
        x = torch.rand(batch_size, sequence_length, d_model)  # Input is now already in d_model dimension

        # Pass input through the Transformer Encoder
        output = transformer_encoder(x)

        # Check if output shape matches the expected shape (batch_size, sequence_length, d_model)
        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    def test_transformer_encoder_layers(self):
        """Test if the TransformerEncoder processes input through all encoder layers."""
        num_layers = 4
        d_model = 16
        num_heads = 4
        d_ff = 64
        max_length = 50
        batch_size = 32
        sequence_length = 20

        # Initialize the Transformer Encoder
        transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_length)
        x = torch.rand(batch_size, sequence_length, d_model)  # Input is now already in d_model dimension

        # Pass input through the Transformer Encoder
        output = transformer_encoder(x)

        # Check that the output is not the same as the input (since it should be transformed by multiple layers)
        self.assertFalse(torch.equal(output, x))

    def test_transformer_encoder_positional_encoding(self):
        """Test if the TransformerEncoder correctly applies positional encoding."""
        num_layers = 4
        d_model = 16
        num_heads = 4
        d_ff = 64
        max_length = 50
        batch_size = 32
        sequence_length = 20

        # Initialize the Transformer Encoder
        transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_length)
        x = torch.rand(batch_size, sequence_length, d_model)  # Input is now already in d_model dimension

        # Apply positional encoding only
        pos_encoded_x = transformer_encoder.pos_encoding(x)

        # Ensure that positional encoding modifies the input
        self.assertFalse(torch.equal(x, pos_encoded_x))

        # Ensure that the shape after positional encoding is correct
        self.assertEqual(pos_encoded_x.shape, (batch_size, sequence_length, d_model))


class TestEncoderLayer(unittest.TestCase):

    def test_encoder_layer_output_shape(self):
        """Test if the EncoderLayer produces the correct output shape."""
        d_model = 16
        num_heads = 4
        d_ff = 64
        batch_size = 32
        sequence_length = 20

        # Initialize the Encoder Layer
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.rand(batch_size, sequence_length, d_model)

        # Pass input through the encoder layer
        output = encoder_layer(x)

        # Check if the output shape matches the input shape
        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    def test_encoder_layer_residual_connection(self):
        """Test if the residual connections are working by modifying the input slightly and checking the output."""
        d_model = 16
        num_heads = 4
        d_ff = 64
        batch_size = 32
        sequence_length = 20

        # Initialize the Encoder Layer
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

        # Create two similar inputs
        x1 = torch.rand(batch_size, sequence_length, d_model)
        x2 = x1.clone()

        # Apply a small perturbation to x2
        x2[:, 0, :] += 0.1

        # Pass both inputs through the encoder layer
        output1 = encoder_layer(x1)
        output2 = encoder_layer(x2)

        # The outputs should be different due to the residual connection and the perturbation
        self.assertFalse(torch.equal(output1, output2))

    def test_encoder_layer_functionality(self):
        """Test the end-to-end functionality of the EncoderLayer."""
        d_model = 16
        num_heads = 4
        d_ff = 64
        batch_size = 32
        sequence_length = 20

        # Initialize the Encoder Layer
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.rand(batch_size, sequence_length, d_model)

        # Pass input through the encoder layer
        output = encoder_layer(x)

        # Ensure the output is not the same as the input
        self.assertFalse(torch.equal(output, x))

        # Ensure that the output has valid (non-NaN) values
        self.assertFalse(torch.isnan(output).any())


class TestInputEmbedding(unittest.TestCase):
    def test_output_shape(self):
        num_inputs = 10
        d_model = 16
        batch_size = 32
        sequence_length = 20

        model = InputEmbedding(num_inputs, d_model)
        x = torch.rand(batch_size, sequence_length, num_inputs)

        output = model(x)

        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    def test_output_changes_with_input(self):
        num_inputs = 10
        d_model = 16
        batch_size = 32
        sequence_length = 20

        model = InputEmbedding(num_inputs, d_model)
        x1 = torch.rand(batch_size, sequence_length, num_inputs)
        x2 = torch.rand(batch_size, sequence_length, num_inputs)

        output1 = model(x1)
        output2 = model(x2)

        self.assertFalse(torch.equal(output1, output2))

    def test_transformation_applied(self):
        num_inputs = 10
        d_model = 16
        batch_size = 32
        sequence_length = 20

        model = InputEmbedding(num_inputs, d_model)
        x = torch.ones(batch_size, sequence_length, num_inputs)  # input is all ones

        output = model(x)

        # Check that the output is not all ones
        self.assertFalse(torch.all(output == 1))


class TestPositionalEncoding(unittest.TestCase):
    def test_output_shape(self):
        d_model = 16
        max_length = 50
        batch_size = 32
        sequence_length = 20

        model = PositionalEncoding(d_model, max_length)
        x = torch.rand(batch_size, sequence_length, d_model)

        output = model(x)

        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    def test_positional_encoding_variation(self):
        d_model = 16
        max_length = 50
        batch_size = 32
        sequence_length = 20

        model = PositionalEncoding(d_model, max_length)
        x = torch.zeros(batch_size, sequence_length, d_model)

        output = model(x)

        # Check that the output is not all zeros, indicating that positional encoding is added
        self.assertFalse(torch.all(output == 0))

    def test_consistency_across_calls(self):
        d_model = 16
        max_length = 50
        batch_size = 32
        sequence_length = 20

        model = PositionalEncoding(d_model, max_length)
        x = torch.rand(batch_size, sequence_length, d_model)

        output1 = model(x)
        output2 = model(x)

        self.assertTrue(torch.equal(output1, output2))


class TestMultiHeadAttention(unittest.TestCase):

    def test_output_shape(self):
        """Test if the MultiHeadAttention layer produces the correct output shape."""
        d_model = 16
        num_heads = 2
        batch_size = 32
        sequence_length = 20

        model = MultiHeadAttention(d_model, num_heads)
        x = torch.rand(batch_size, sequence_length, d_model)

        output = model(x,x,x)

        # Check if output shape matches expected shape (batch_size, sequence_length, d_model)
        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    def test_attention_scores(self):
        """Test if the attention scores have the correct shape and are computed properly."""
        d_model = 16
        num_heads = 4
        batch_size = 32
        sequence_length = 20

        model = MultiHeadAttention(d_model, num_heads)
        x = torch.rand(batch_size, sequence_length, d_model)

        # Get query and key matrices
        Q = model.query(x)
        K = model.key(x)

        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)
        K = K.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_model // num_heads, dtype=torch.float32))

        # Check if attention scores shape matches expected shape (batch_size, num_heads, sequence_length, sequence_length)
        self.assertEqual(attention_scores.shape, (batch_size, num_heads, sequence_length, sequence_length))

        # Ensure that the attention scores are not all the same (indicating the attention is active)
        self.assertFalse(torch.all(attention_scores == attention_scores[0, 0, 0, 0]))

    def test_different_input_sizes(self):
        """Test if MultiHeadAttention layer works with varying sequence lengths."""
        d_model = 16
        num_heads = 4

        model = MultiHeadAttention(d_model, num_heads)

        for seq_len in [10, 20, 30]:
            x = torch.rand(32, seq_len, d_model)
            output = model(x,x,x)

            # Check if output shape matches expected shape (batch_size, sequence_length, d_model)
            self.assertEqual(output.shape, (32, seq_len, d_model))

    def test_attention_masking(self):
        """Test that the attention mechanism correctly applies the mask."""
        d_model = 16
        num_heads = 4
        attention = MultiHeadAttention(d_model, num_heads)
        batch_size = 2
        seq_len = 5

        # Create a batch of random input sequences
        x = torch.rand(batch_size, seq_len, d_model)

        # Create a target mask that blocks future tokens
        mask = generate_target_mask(seq_len)

        # Pass the mask and input through the attention mechanism
        attention_output = attention(x, x, x, mask)

        # Compute the attention scores manually
        Q = attention.query(x)
        K = attention.key(x)
        V = attention.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, attention.num_heads, attention.dk).transpose(1, 2)
        K = K.view(batch_size, seq_len, attention.num_heads, attention.dk).transpose(1, 2)
        V = V.view(batch_size, seq_len, attention.num_heads, attention.dk).transpose(1, 2)

        # Manually compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(attention.dk, dtype=torch.float32))

        # Apply the mask manually
        # The mask should have the shape [batch_size, num_heads, seq_len, seq_len] to match attention_scores
        mask = mask.unsqueeze(0).unsqueeze(0)  # Reshape to [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, num_heads, -1, -1)  # Expand to [batch_size, num_heads, seq_len, seq_len]

        # Apply the mask by setting positions where mask == True to -inf
        attention_scores_masked = attention_scores.masked_fill(mask == True, float('-inf'))

        # Apply softmax to the masked attention scores
        attention_weights = torch.nn.functional.softmax(attention_scores_masked, dim=-1)

        # Compute the manually masked attention output
        manual_attention_output = torch.matmul(attention_weights, V)
        manual_attention_output = manual_attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                            d_model)

        # Pass the manual masked output through the final linear layer to match the MultiHeadAttention output
        manual_attention_output = attention.out(manual_attention_output)

        # Check if the attention output from the class matches the manually calculated one
        self.assertTrue(torch.allclose(attention_output, manual_attention_output, atol=1e-6))


class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        d_model = 16
        num_heads = 4
        d_ff = 32
        self.decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = d_model

    def test_forward(self):
        """Test the forward pass of the DecoderLayer."""
        # Create random input sequences (batch_size, seq_len, d_model)
        x = torch.rand(self.batch_size, self.seq_len, self.d_model)
        encoder_output = torch.rand(self.batch_size, self.seq_len, self.d_model)

        # Generate a target mask (seq_len, seq_len)
        tgt_mask = generate_target_mask(self.seq_len)

        # Pass the input through the decoder layer
        output = self.decoder_layer(x, encoder_output, tgt_mask)

        # Ensure the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_masking(self):
        """Test that the target mask is applied correctly."""
        x = torch.rand(self.batch_size, self.seq_len, self.d_model)
        encoder_output = torch.rand(self.batch_size, self.seq_len, self.d_model)

        # Create a target mask
        tgt_mask = generate_target_mask(self.seq_len)

        # Pass the input through the decoder layer
        output = self.decoder_layer(x, encoder_output, tgt_mask)

        # Manually check the masking by examining attention scores
        # This is more of a sanity check to ensure the mask is being applied
        self.assertTrue(output is not None)  # Basic check to ensure output is computed


class TestTransformerDecoder(unittest.TestCase):
    def setUp(self):
        num_layers = 2
        d_model = 16
        num_heads = 4
        d_ff = 32
        dropout = 0.1
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = d_model

    def test_forward(self):
        """Test the forward pass of the TransformerDecoder."""
        # Create random input sequences (batch_size, seq_len, d_model)
        x = torch.rand(self.batch_size, self.seq_len, self.d_model)
        encoder_output = torch.rand(self.batch_size, self.seq_len, self.d_model)

        # Generate a target mask
        tgt_mask = generate_target_mask(self.seq_len)

        # Pass the input through the decoder
        output = self.decoder(x, encoder_output, tgt_mask)

        # Ensure the output has the correct shape
        self.assertEqual(output.shape, (
        self.batch_size, self.seq_len, 1))  # The last dimension is 1 since we're predicting continuous values

    def test_inference(self):
        """Test the inference method of the TransformerDecoder."""
        # Create a random encoder output (batch_size, seq_len, d_model)
        encoder_output = torch.rand(self.batch_size, self.seq_len, self.d_model)

        # Use a random start token with correct shape
        start_token = torch.zeros(self.d_model)

        # Define max_len for inference (how many steps to predict)
        max_len = 3

        # Generate predictions
        predictions = self.decoder.inference(encoder_output, start_token, max_len)

        # Ensure the predictions have the correct shape
        self.assertEqual(predictions.shape, (self.batch_size, max_len,1))

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Set up models parameters
        self.num_layers = 2
        self.d_model = 16
        self.num_heads = 4
        self.d_ff = 32
        self.input_dim = 10
        self.output_dim = 1
        self.max_len = 5

        # Initialize the Transformer models
        self.model = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            encoder_input_dim=self.input_dim,
            decoder_input_dim=self.output_dim,
            dropout=0.1
        )

        # Set up dummy data for testing
        self.batch_size = 2
        self.encoder_input = torch.rand(self.batch_size, self.max_len, self.input_dim)
        self.decoder_input = torch.rand(self.batch_size, self.max_len, self.output_dim)
        self.start_token = torch.zeros(self.d_model)

    def test_forward(self):
        """Test the forward pass of the Transformer models."""
        tgt_mask = generate_target_mask(self.decoder_input.size(1))

        # Forward pass
        output = self.model(self.encoder_input, self.decoder_input, tgt_mask)

        # Check the shape of the output
        self.assertEqual(output.shape, (self.batch_size, self.max_len, self.output_dim))

    def test_inference(self):
        """Test the inference method of the Transformer models."""
        predictions = self.model.inference(self.encoder_input, self.start_token, self.max_len)

        # Check the shape of the predictions

        self.assertEqual(predictions.shape, torch.Size([self.batch_size, self.max_len, 1]))


class TestContinuousLoss(unittest.TestCase):

    def test_continuous_loss_full_sequence(self):
        # Case 1: Model predicts the full sequence correctly with continuation signal > 0.5
        y_pred = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        y_true = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        continuation_signals = torch.tensor([[0.9, 0.9, 0.9, 0.9]])

        loss = continuous_loss(y_pred, y_true, continuation_signals, lambda_penalty=1.0)
        expected_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        self.assertTrue(torch.isclose(loss, expected_loss), f"Expected loss {expected_loss}, but got {loss}")

    def test_continuous_loss_early_stop(self):
        # Case 2: Model stops early (only predicts the first 2 steps)
        y_pred = torch.tensor([[[1.0], [2.0], [0.0], [0.0]]])
        y_true = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        continuation_signals = torch.tensor([[0.9, 0.4, 0.0, 0.0]])

        loss = continuous_loss(y_pred, y_true, continuation_signals, lambda_penalty=1.0)
        expected_loss = F.mse_loss(y_pred[:, :2, :], y_true[:, :2, :], reduction='mean') + 3
        self.assertTrue(torch.isclose(loss, expected_loss), f"Expected loss {expected_loss}, but got {loss}")

    def test_continuous_loss_stop_after_first(self):
        # Case 3: Model predicts the first step and then stops immediately
        y_pred = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
        y_true = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        continuation_signals = torch.tensor([[0.4, 0.0, 0.0, 0.0]])

        loss = continuous_loss(y_pred, y_true, continuation_signals, lambda_penalty=1.0)
        expected_loss = torch.tensor(4.0)
        self.assertTrue(torch.isclose(loss, expected_loss), f"Expected loss {expected_loss}, but got {loss}")

    def test_continuous_loss_full_sequence_with_over_prediction(self):
        # Case 4: Model predicts more steps than needed (all continuation signals > 0.5)
        y_pred = torch.tensor([[[1.0], [2.0], [3.0], [5.0]]])
        y_true = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        continuation_signals = torch.tensor([[0.9, 0.9, 0.9, 0.9]])

        loss = continuous_loss(y_pred, y_true, continuation_signals, lambda_penalty=1.0)
        expected_loss = F.mse_loss(y_pred[:, :4, :], y_true[:, :4, :], reduction='mean')
        self.assertTrue(torch.isclose(loss, expected_loss), f"Expected loss {expected_loss}, but got {loss}")


class TestChannelWiseMultiHeadAttention(unittest.TestCase):

    def test_output_shape(self):
        """Test if the ChannelWiseMultiHeadAttention layer produces the correct output shape."""
        d_model = 16
        num_heads = 2
        batch_size = 32
        sequence_length = 20
        num_features = d_model

        model = ChannelWiseMultiHeadAttention(d_model, num_heads)
        x = torch.rand(batch_size, sequence_length, num_features)

        output = model(x, x, x)

        # Check if output shape matches expected shape (batch_size, sequence_length, num_features)
        self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

    class TestChannelWiseMultiHeadAttention(unittest.TestCase):

        def test_output_shape(self):
            """Test if the ChannelWiseMultiHeadAttention layer produces the correct output shape."""
            d_model = 16
            num_heads = 2
            batch_size = 32
            sequence_length = 20

            model = ChannelWiseMultiHeadAttention(d_model, num_heads)
            x = torch.rand(batch_size, sequence_length, d_model)

            output = model(x, x, x)

            # Check if output shape matches expected shape (batch_size, sequence_length, d_model)
            self.assertEqual(output.shape, (batch_size, sequence_length, d_model))

        def test_attention_scores(self):
            """Test if the attention scores have the correct shape and are computed properly."""
            d_model = 16
            num_heads = 4
            batch_size = 32
            sequence_length = 20

            model = ChannelWiseMultiHeadAttention(d_model, num_heads)
            x = torch.rand(batch_size, sequence_length, d_model)

            # Get query and key matrices
            Q = model.query(x)
            K = model.key(x)

            # Permute for channel-wise attention (batch_size, d_model, seq_len)
            Q = Q.permute(0, 2, 1)
            K = K.permute(0, 2, 1)

            # Reshape for multi-head attention
            Q = Q.view(batch_size, num_heads, d_model // num_heads, sequence_length).transpose(1, 2)
            K = K.view(batch_size, num_heads, d_model // num_heads, sequence_length).transpose(1, 2)

            # Calculate attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(d_model // num_heads, dtype=torch.float32))

            # Check if attention scores shape matches expected shape (batch_size, num_heads, d_model // num_heads, d_model // num_heads)
            self.assertEqual(attention_scores.shape,
                             (batch_size, num_heads, d_model // num_heads, d_model // num_heads))

        def test_different_input_sizes(self):
            """Test if ChannelWiseMultiHeadAttention layer works with varying sequence lengths."""
            d_model = 16
            num_heads = 4

            model = ChannelWiseMultiHeadAttention(d_model, num_heads)

            for seq_len in [10, 20, 30]:
                x = torch.rand(32, seq_len, d_model)
                output = model(x, x, x)

                # Check if output shape matches expected shape (batch_size, sequence_length, d_model)
                self.assertEqual(output.shape, (32, seq_len, d_model))

        def test_attention_masking(self):
            """Test that the attention mechanism correctly applies the mask."""
            d_model = 16
            num_heads = 4
            attention = ChannelWiseMultiHeadAttention(d_model, num_heads)
            batch_size = 2
            seq_len = 5
            num_features = d_model

            # Create a batch of random input sequences
            x = torch.rand(batch_size, seq_len, num_features)

            # Create a mask (for example, upper triangular to block future tokens)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

            # Pass the mask and input through the attention mechanism
            attention_output = attention(x, x, x, mask)

            # Manually compute attention scores
            Q = attention.query(x).permute(0, 2, 1)
            K = attention.key(x).permute(0, 2, 1)
            V = attention.value(x).permute(0, 2, 1)

            # Reshape for multi-head attention
            Q = Q.view(batch_size, num_features, num_heads, attention.dk).transpose(1, 2)
            K = K.view(batch_size, num_features, num_heads, attention.dk).transpose(1, 2)
            V = V.view(batch_size, num_features, num_heads, attention.dk).transpose(1, 2)

            # Manually compute attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(attention.dk, dtype=torch.float32))

            # Apply the mask manually
            mask = mask.unsqueeze(0).unsqueeze(0)  # Reshape to [1, 1, seq_len, seq_len]
            mask = mask.expand(batch_size, num_heads, -1, -1)  # Expand to [batch_size, num_heads, seq_len, seq_len]
            attention_scores_masked = attention_scores.masked_fill(mask == True, float('-inf'))

            # Apply softmax to the masked attention scores
            attention_weights = torch.nn.functional.softmax(attention_scores_masked, dim=-1)

            # Compute the manually masked attention output
            manual_attention_output = torch.matmul(attention_weights, V)
            manual_attention_output = manual_attention_output.transpose(1, 2).contiguous().view(batch_size,
                                                                                                num_features,
                                                                                                seq_len)
            manual_attention_output = manual_attention_output.permute(0, 2, 1)

            # Pass the manual masked output through the final linear layer to match the ChannelWise attention output
            manual_attention_output = attention.out(manual_attention_output)

            # Check if the attention output from the class matches the manually calculated one
            self.assertTrue(torch.allclose(attention_output, manual_attention_output, atol=1e-6))