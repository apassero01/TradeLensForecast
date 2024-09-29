import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import stats
import statsmodels.api as sm
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def layer_heatmap(output_dict, idx):
    """
    This function creates subplots of heatmaps for each attention map in the dictionary.

    Args:
    - component_dict (dict): A dictionary where keys are component names and values are attention maps (numpy arrays).
    - idx (int): The index to extract the specific attention map from the value array.
    """

    num_components = 0
    for key in output_dict.keys():
        num_components += len(output_dict[key])

    num_cols = 1  # You can adjust this depending on how many plots you want per row
    num_rows = (num_components + num_cols - 1) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * num_rows))

    # vmin = -1
    # vmax = 1

    i = 0
    for key in output_dict.keys():
        component_dict = output_dict[key]
        for item in (component_dict.keys()):
            attention_weights = component_dict[item]
            # Extracting a single set of attention weights for visualization
            attention_weights_single = attention_weights[idx] if len(attention_weights.shape) > 2 else attention_weights
            attention_weights_single = attention_weights_single.cpu().detach().numpy().T

            ax = plt.subplot(num_rows, num_cols, i + 1)
            norm = None  # You can set the norm if you want to standardize color scales

            # sns.heatmap(attention_weights_single, cmap='viridis', linewidths=0.5, cbar=True, norm=norm, ax=ax, vmin=vmin, vmax=vmax)
            sns.heatmap(attention_weights_single, cmap='viridis', linewidths=0.5, cbar=True, norm=norm, ax=ax)


            ax.set_title(f'{key} - {item}')
            ax.set_ylabel('Output Sequence Position')
            ax.set_xlabel('Input Sequence Position')

            # Set min max for colorbar if needed
            ax.set_xticks(range(len(attention_weights_single[0])))
            ax.set_xticklabels(range(1, len(attention_weights_single[0]) + 1), rotation=45)

            ax.set_yticks(range(len(attention_weights_single)))
            ax.set_yticklabels(range(1, len(attention_weights_single) + 1))
            i+=1

    plt.tight_layout()
    plt.show()


def visualize_future_predictions_combined(true_steps, pred_steps, alreadyCum=False):
    plt.figure(figsize=(14, 6))

    # First subplot: original values
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(true_steps, label='True', color='blue')
    plt.plot(pred_steps, label='Predicted', color='orange')
    plt.title('Future Predictions')
    plt.legend()
    plt.ylim(-6, 6)

    # Second subplot: cumulative sum
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    if not alreadyCum:
        plt.plot(np.cumsum(true_steps), label='True (Cumulative)', color='blue')
        plt.plot(np.cumsum(pred_steps), label='Predicted (Cumulative)', color='orange')
    else :
        plt.plot(true_steps, label='True (Cumulative)', color='blue')
        plt.plot(pred_steps, label='Predicted (Cumulative)', color='orange')
    plt.title('Cumulative Future Predictions')
    plt.legend()
    plt.ylim(-6,6)

    plt.tight_layout()
    plt.show()

def visualize_future_predictions_with_uncertainty(
    true_steps, mu_pred, std_pred, already_cumulative=False
):
    """
    Visualizes the true values, predicted mean, and confidence intervals.

    Args:
        true_steps (array-like): The true target values.
        mu_pred (array-like): The predicted mean values.
        std_pred (array-like): The predicted standard deviations.
        already_cumulative (bool): If True, data is already cumulative.
    """
    time_steps = np.arange(len(true_steps))

    # Compute confidence intervals (e.g., 95% confidence interval)
    confidence_coefficient = 1.96  # For 95% CI
    upper_bound = mu_pred + confidence_coefficient * std_pred
    lower_bound = mu_pred - confidence_coefficient * std_pred

    plt.figure(figsize=(14, 6))

    # First subplot: Predicted mean with confidence intervals
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, true_steps, label='True', color='blue')
    plt.plot(time_steps, mu_pred, label='Predicted Mean', color='orange')
    plt.fill_between(
        time_steps,
        lower_bound,
        upper_bound,
        color='gray',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    plt.title('Future Predictions with Uncertainty')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)

    # Second subplot: Cumulative sum (if applicable)
    plt.subplot(1, 2, 2)
    if not already_cumulative:
        true_cumulative = np.cumsum(true_steps)
        pred_cumulative = np.cumsum(mu_pred)
        upper_cumulative = np.cumsum(upper_bound)
        lower_cumulative = np.cumsum(lower_bound)
    else:
        true_cumulative = true_steps
        pred_cumulative = mu_pred
        upper_cumulative = upper_bound
        lower_cumulative = lower_bound

    plt.plot(time_steps, true_cumulative, label='True (Cumulative)', color='blue')
    plt.plot(time_steps, pred_cumulative, label='Predicted Mean (Cumulative)', color='orange')
    plt.fill_between(
        time_steps,
        lower_cumulative,
        upper_cumulative,
        color='gray',
        alpha=0.3,
        label='95% Confidence Interval (Cumulative)'
    )
    plt.title('Cumulative Future Predictions with Uncertainty')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_gradients(gradients_with_names, max_grad_norm=None):
    """
    Visualize the gradients of a models's parameters.

    Args:
    - gradients_with_names: List of tuples containing (layer_name, gradient_tensor).
    - max_grad_norm: Optional maximum gradient norm for scaling.
    """
    # Unpack layer names and gradients from the input list
    layer_names = [name for name, _ in gradients_with_names]
    grad_norms = [grad.norm().item() for _, grad in gradients_with_names]

    # Normalize gradients if a max norm is provided
    if max_grad_norm:
        grad_norms = [min(grad, max_grad_norm) for grad in grad_norms]

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(layer_names, grad_norms, color='blue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms of Model Parameters')
    plt.ylim(0, max(grad_norms) * 1.1)
    plt.show()



def visualize_step_distribution(y_pred, y):
    flattened_X = y_pred.flatten()
    flattened_Y = y.flatten()

    # Create the side-by-side box plots
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the box plots
    ax.boxplot([flattened_X, flattened_Y], notch=False, patch_artist=True, labels=['y_pred', 'y_true'])

    # Set titles and labels
    ax.set_title('Side-by-Side Box Plots for X and Y')
    ax.set_ylabel('Values')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(data, features, feature_dict):
    """
    Plots histograms for each feature specified in feature_dict from a 3D NumPy array.

    Parameters:
    - data (np.ndarray): A 3D NumPy array of shape (batch_size, seq_length, num_features).
    - feature_dict (dict): A dictionary where keys are feature names and values are indices in the 3rd dimension of the data array.
    """
    num_features = len(features)

    # Determine the grid size for subplots
    nrows = 4
    ncols = (num_features + nrows - 1) // nrows

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over the feature dictionary
    for i, feature_name in enumerate(features):
        # Flatten the data for the given feature index across all batches and sequences
        feature_index = feature_dict[feature_name]
        feature_data = data[:, :, feature_index].flatten()

        # Plot the histogram on the corresponding subplot
        axes[i].hist(feature_data, bins=30, edgecolor='black')
        axes[i].set_title(feature_name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Turn off any empty subplots (if num_features is not a perfect multiple of nrows * ncols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_y_feature_histograms(data, features):
    """
    Plots histograms for each feature specified in feature_dict from a 3D NumPy array.

    Parameters:
    - data (np.ndarray): A 3D NumPy array of shape (batch_size, seq_length, num_features).
    - feature_dict (dict): A dictionary where keys are feature names and values are indices in the 3rd dimension of the data array.
    """
    num_features = len(features)

    # Determine the grid size for subplots
    nrows = 4
    ncols = (num_features + nrows - 1) // nrows
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over the feature dictionary
    for i, feature_name in enumerate(features):
        # Flatten the data for the given feature index across all batches and sequences
        feature_index = i
        feature_data = data[:, feature_index, : ].flatten()

        # Plot the histogram on the corresponding subplot
        axes[i].hist(feature_data, bins=30, edgecolor='black')
        axes[i].set_title(feature_name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Turn off any empty subplots (if num_features is not a perfect multiple of nrows * ncols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


def visualize_std(std_pred):
    """
    Visualize the variance (standard deviation predictions) for each step using box plots.

    Parameters:
    - std_pred: numpy array of shape (elements, output_steps), where each element represents
      a predicted distribution's standard deviation for future time steps.
    """
    elements, output_steps = std_pred.shape

    # Create a figure
    plt.figure(figsize=(14, 6))

    # Create a box plot for each step in the prediction horizon
    for step in range(output_steps):
        plt.boxplot(std_pred[:, step], positions=[step], widths=0.6, showmeans=True)

    # Set labels
    plt.title('Predicted Variance (Standard Deviation) Across Future Steps')
    plt.xlabel('Prediction Step')
    plt.ylabel('Predicted Standard Deviation')

    # Set xticks to represent the prediction steps
    plt.xticks(range(output_steps), [f'Step {i}' for i in range(output_steps)])

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred_mean, y_pred_std):
    """
    Plot various residual-related plots in a grid: residuals over time, mean residuals, error histogram,
    and Q-Q plot to check the normality of the residuals.

    Parameters:
    - y_true: numpy array of true values, shape (elements, output_steps)
    - y_pred_mean: numpy array of predicted mean values, shape (elements, output_steps)
    - y_pred_std: numpy array of predicted standard deviations, shape (elements, output_steps)
    """
    residuals = y_true - y_pred_mean
    mean_residuals = residuals.mean(axis=0)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals plot over time (for all elements)
    for element in range(residuals.shape[0]):
        axs[0, 0].plot(residuals[element], marker='o', linestyle='-', alpha=0.5)
    axs[0, 0].axhline(0, color='black', linewidth=1)
    axs[0, 0].set_title('Residuals Across Entire Dataset Over Time')
    axs[0, 0].set_xlabel('Time Steps')
    axs[0, 0].set_ylabel('Residuals (True - Predicted Mean)')

    # 2. Mean residuals plot over time
    axs[0, 1].plot(mean_residuals, marker='o', linestyle='-', color='red')
    axs[0, 1].axhline(0, color='black', linewidth=1)
    axs[0, 1].set_title('Mean Residuals Over Time')
    axs[0, 1].set_xlabel('Time Steps')
    axs[0, 1].set_ylabel('Mean Residuals')

    # 3. Error histogram
    axs[1, 0].hist(residuals.flatten(), bins=30, color='gray', edgecolor='black')
    axs[1, 0].set_title('Error Histogram')
    axs[1, 0].set_xlabel('Residuals (True - Predicted Mean)')
    axs[1, 0].set_ylabel('Frequency')

    # 4. Q-Q plot to check if residuals are normally distributed
    sm.qqplot(residuals.flatten(), line='s', ax=axs[1, 1])
    axs[1, 1].set_title('Q-Q Plot of Residuals')

    plt.tight_layout()
    plt.show()


def plot_layer_activations(model, encoder_input):
    """
    Visualize the activations of each layer in the transformer.

    Parameters:
    - models: The transformer models (should have an encoder and projection layers)
    - encoder_input: The raw input data to be projected and passed through the transformer encoder
    """
    with torch.no_grad():
        # 1. Apply input projection to transform the encoder input to the correct dimension
        x = model.input_projection(encoder_input)

        # 2. Apply positional encoding (required before passing into encoder layers)
        x = model.encoder.pos_encoding(x)

        # 3. Collect activations after each encoder layer
        layer_outputs = []
        for layer in model.encoder.encoder_layers:
            x = layer(x)
            layer_outputs.append(x.cpu().numpy())  # Store activations for visualization

    # 4. Plot activations layer by layer
    fig, axes = plt.subplots(len(layer_outputs), 1, figsize=(10, len(layer_outputs) * 2))
    for i, layer_output in enumerate(layer_outputs):
        mean_activations = np.mean(layer_output, axis=-1)  # Calculate mean activations across the batch
        axes[i].hist(mean_activations.flatten(), bins=30)
        axes[i].set_title(f'Layer {i + 1} Activations')

    plt.tight_layout()
    plt.show()


def extract_embeddings(model, encoder_input):
    """
    Extract the embeddings from the transformer's input projection and positional encoding layers.

    Parameters:
    - models: The transformer models
    - encoder_input: Raw input data (before being passed through the transformer)

    Returns:
    - embeddings: The projected and positionally encoded embeddings
    """
    with torch.no_grad():
        # 1. Project the input to d_model dimension using the input projection layer
        embeddings = model.input_projection(encoder_input)

        # 2. Apply positional encoding to the projected embeddings
        embeddings = model.encoder.pos_encoding(embeddings)

    return embeddings.cpu().numpy()


def visualize_embeddings(embeddings, method='tsne', perplexity=30):
    """
    Visualize the embeddings using dimensionality reduction (t-SNE or PCA).

    Parameters:
    - embeddings: The extracted embeddings, shape (batch_size, sequence_length, d_model)
    - method: 'tsne' or 'pca' for dimensionality reduction
    - perplexity: t-SNE perplexity parameter (optional)
    """
    # Flatten embeddings from (batch_size, sequence_length, d_model) to (batch_size * sequence_length, d_model)
    flattened_embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    if method == 'pca':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(flattened_embeddings)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(flattened_embeddings)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    # Plot the reduced embeddings
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, cmap='Spectral')
    plt.title(f'Embeddings Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()