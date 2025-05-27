import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    proj_weight = nodes[0]  
    X = ad.matmul(X, proj_weight)  

    Q = ad.matmul(X, nodes[1])
    K = ad.matmul(X, nodes[2])
    V = ad.matmul(X, nodes[3])

    K_T = ad.transpose(K, -2, -1) 
    scores = ad.matmul(Q, K_T)
    scores = scores / (model_dim ** 0.5)
    
    attn_weights = ad.softmax(scores, dim=-1)
    
    attended = ad.matmul(attn_weights, V)
    
    out = ad.matmul(attended, nodes[4])
    
    out = ad.layernorm(out, [model_dim], eps)
    
    ff_out = ad.matmul(out, nodes[5]) + nodes[6]
    ff_out = ad.relu(ff_out)
    ff_out = ad.matmul(ff_out, nodes[7]) + nodes[8]
    
    output = ad.mean(ff_out, dim=1)
    
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).
    """
    probs = ad.softmax(Z, dim=-1)
    
    log_probs = ad.log(probs)
    
    loss = ad.mul(y_one_hot, log_probs)
    loss = ad.sum_op(loss, dim=1)
    total_loss = ad.sum_op(loss, dim=0)
    
    return ad.mul_by_const(total_loss, -1.0/batch_size)


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the transformer model."""

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        output = f_run_model(X_batch, y_batch, model_weights)
        logits, loss, *gradients = output
        
        for j, (weight, grad) in enumerate(zip(model_weights, gradients)):
            if grad.dim() > weight.dim():
                sum_dims = tuple(range(grad.dim() - weight.dim()))
                grad = torch.sum(grad, dim=sum_dims)
            
            for i, (g_size, w_size) in enumerate(zip(grad.shape[::-1], weight.shape[::-1])):
                if g_size != w_size:
                    if g_size == 1:
                        grad = grad.expand_as(weight)
                    else:
                        grad = torch.sum(grad, dim=len(grad.shape)-1-i, keepdim=True)
            
            model_weights[j] = weight - lr * grad
            
        total_loss += loss.item()
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)
    
    return model_weights, average_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    
    # Set up model params
    # Hyperparameters
    input_dim = 28  
    seq_length = max_len
    num_classes = 10
    model_dim = 128
    eps = 1e-5
    
    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02
    
    # Create input nodes
    X = ad.Variable(name="X")
    
    # Define the forward graph.
    W_proj = ad.Variable(name="W_proj")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    b_1 = ad.Variable(name="b_1")
    W_2 = ad.Variable(name="W_2")
    b_2 = ad.Variable(name="b_2")
    
    nodes = [W_proj, W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2]
    
    # Define forward computation graph
    y_predict: ad.Node = transformer(X, nodes, model_dim, seq_length, eps, batch_size, num_classes)
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # Create the evaluator
    grads: List[ad.Node] = ad.gradients(loss, nodes)
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])
    
    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    
    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()
    
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    
    num_classes = 10

    # Initialize model weights
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(model_dim)
    W_proj_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_Q_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        return evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                W_proj: model_weights[0],
                W_Q: model_weights[1],
                W_K: model_weights[2],
                W_V: model_weights[3],
                W_O: model_weights[4],
                W_1: model_weights[5],
                b_1: model_weights[6],
                W_2: model_weights[7],
                b_2: model_weights[8]
            }
        )
    
    def f_eval_model(X_val, model_weights):
        """Evaluate model on validation/test data"""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            
            logits = test_evaluator.run({
                X: torch.tensor(X_batch, dtype=torch.float64),
                W_proj: model_weights[0],
                W_Q: model_weights[1],
                W_K: model_weights[2],
                W_V: model_weights[3],
                W_O: model_weights[4],
                W_1: model_weights[5],
                b_1: model_weights[6],
                W_2: model_weights[7],
                b_2: model_weights[8]
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions
    
    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    # Initialize the model weights here
    model_weights = [
        torch.tensor(W_proj_val, dtype=torch.float64),
        torch.tensor(W_Q_val, dtype=torch.float64),
        torch.tensor(W_K_val, dtype=torch.float64),
        torch.tensor(W_V_val, dtype=torch.float64),
        torch.tensor(W_O_val, dtype=torch.float64),
        torch.tensor(W_1_val, dtype=torch.float64),
        torch.tensor(b_1_val, dtype=torch.float64),
        torch.tensor(W_2_val, dtype=torch.float64),
        torch.tensor(b_2_val, dtype=torch.float64)
    ]

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )
        
        # Evaluate on test data
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test.numpy())}, "
            f"loss = {loss_val}"
        )
    
    # Final evaluation
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
