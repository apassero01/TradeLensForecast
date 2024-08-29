import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_attention_output_multi(component_dict, idx):
    """
    This function creates subplots of heatmaps for each attention map in the dictionary.

    Args:
    - component_dict (dict): A dictionary where keys are component names and values are attention maps (numpy arrays).
    - idx (int): The index to extract the specific attention map from the value array.
    """
    num_components = len(component_dict)
    num_cols = 1  # You can adjust this depending on how many plots you want per row
    num_rows = (num_components + num_cols - 1) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * num_rows))

    vmin = -1
    vmax = 1

    for i, (component_name, attention_weights) in enumerate(component_dict.items()):
        # Extracting a single set of attention weights for visualization
        attention_weights_single = attention_weights[idx] if len(attention_weights.shape) > 2 else attention_weights
        attention_weights_single = attention_weights_single.cpu().detach().numpy()

        ax = plt.subplot(num_rows, num_cols, i + 1)
        norm = None  # You can set the norm if you want to standardize color scales

        sns.heatmap(attention_weights_single, cmap='viridis', linewidths=0.5, cbar=True, norm=norm, ax=ax, vmin=vmin, vmax=vmax)

        ax.set_title(f'Attention Weights: {":".join(component_name.split(":")[-2:])}')
        ax.set_ylabel('Output Sequence Position')
        ax.set_xlabel('Input Sequence Position')

        # Set min max for colorbar if needed
        ax.set_xticks(range(len(attention_weights_single[0])))
        ax.set_xticklabels(range(1, len(attention_weights_single[0]) + 1), rotation=45)

        ax.set_yticks(range(len(attention_weights_single)))
        ax.set_yticklabels(range(1, len(attention_weights_single) + 1))

    plt.tight_layout()
    plt.show()


def visualize_future_predictions_combined(true_steps, pred_steps):
    plt.figure(figsize=(14, 6))

    # First subplot: original values
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(true_steps, label='True', color='blue')
    plt.plot(pred_steps, label='Predicted', color='orange')
    plt.title('Future Predictions')
    plt.legend()
    plt.ylim(-12, 12)

    # Second subplot: cumulative sum
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(np.cumsum(true_steps), label='True (Cumulative)', color='blue')
    plt.plot(np.cumsum(pred_steps), label='Predicted (Cumulative)', color='orange')
    plt.title('Cumulative Future Predictions')
    plt.legend()
    plt.ylim(-12, 12)

    plt.tight_layout()
    plt.show()

def visualize_gradients(gradients_with_names, max_grad_norm=None):
    """
    Visualize the gradients of a model's parameters.

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