Samarth Banodia , 24B0392
# 🤖 AI Agents in LangGraph

#### 🚀 What I Have Learned So Far

##### 1️ PyTorch and Neural Networks

PyTorch as an open-source deep learning library with GPU acceleration, dynamic computational graphs, and automatic differentiation (autograd).

Tensors: multi-dimensional arrays used for input data, model parameters, and gradients.

Autograd for automatic backpropagation.

Training pipeline: dataset loading, preprocessing, forward pass, loss calculation, backpropagation, and parameter updates.

Using torch.nn for layers (Linear, Conv2D, LSTM), activation functions (ReLU, Sigmoid, Tanh), and loss functions (CrossEntropyLoss, MSELoss).

Using torch.optim for optimizers with built-in weight decay (L2 regularization).

Efficient data handling with Dataset and DataLoader, enabling mini-batch gradient descent, shuffling, parallel loading, and custom samplers.

GPU training workflows, enabling speedups with pinned memory and larger batch sizes.

Overfitting handling: dropout, L2 regularization, batch normalization, early stopping, and data augmentation.

Hyperparameter tuning using Optuna for systematic experimentation.

Building and training CNNs for image data and RNNs/LSTMs for sequential data.

Understanding concepts like internal covariate shift, batch normalization internals, and transfer learning with pre-trained models.

##### 2️ Hugging Face

Understanding Hugging Face as a platform for accessing pre-trained models and datasets.

Model Hub: using from_pretrained to load pre-trained models for various NLP tasks.

Tokenizers: handling text pre-processing, tokenization, and encoding for models.

Pipelines: simplified workflows for tasks like sentiment analysis, summarization, and translation using pipeline().

Saving and loading models using model.save_pretrained() and model.from_pretrained() for fine-tuned models.

Leveraging Hugging Face datasets for structured workflows during experimentation.

##### 3️ Introduction to LangGraph

LangGraph enables structured graph-based workflows for LLMs, allowing for composable, multi-step reasoning pipelines.

Understanding graph nodes as LLM-powered units, with edges dictating flow based on outputs.

Potential use cases include tool-augmented agents, document Q&A systems, and stateful reasoning chains.

POA from week 5 onwards : ![image](https://github.com/user-attachments/assets/b2cf6a3a-3c1d-41bd-b955-4faa42b765bd)
