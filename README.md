# Deep Learning for Sequential Data

This repository showcases a series of deep learning projects focused on processing sequential data, implemented as part of a university-level Deep Learning course (FIT5215: Deep Learning 2024). The projects demonstrate fundamental concepts and advanced architectures, including Recurrent Neural Networks (RNNs) and Transformer models, primarily using PyTorch.

## Project Overview

This repository contains three Jupyter Notebooks, each focusing on a specific aspect of deep learning for sequential data:

### 1. `01_RNN_Fundamentals.ipynb`
This notebook introduces the foundational concepts of Recurrent Neural Networks. It includes the manual implementation of a multi-timestep Recurrent Neural Network designed for classification tasks. This section covers the core mechanics of RNNs and their application to basic sequence modeling problems.

**Key Learnings/Skills Demonstrated:**
* Understanding of RNN architecture and forward/backward passes.
* Manual implementation of custom RNN layers.
* Handling sequential data for classification.

### 2. `02_RNN_for_Text_Classification.ipynb`
Building upon the fundamentals, this notebook delves into applying Recurrent Neural Networks (including advanced variants like GRU/LSTM if applicable in the original assignment) for text classification. It features a robust data preprocessing pipeline, including tokenization using `BertTokenizer`, numericalization, padding, and efficient data loading. The project applies RNNs to a question classification dataset.

**Key Learnings/Skills Demonstrated:**
* Text data preprocessing pipelines (tokenization, padding, numericalization).
* Using `BertTokenizer` for effective text preparation.
* Implementing and training RNN models for natural language processing (NLP) tasks.
* Proficiency with PyTorch for model development and training.

### 3. `03_Transformers_for_NLP.ipynb`
This notebook explores the powerful Transformer architecture, a cornerstone of modern NLP. It covers the principles of self-attention and how Transformers are applied to sequence-to-sequence tasks, potentially involving pre-trained models like BERT for various NLP challenges.

**Key Learnings/Skills Demonstrated:**
* Understanding the Transformer architecture and attention mechanisms.
* Working with pre-trained Transformer models (e.g., BERT) using libraries like `Hugging Face Transformers`.
* Applying state-of-the-art models to advanced NLP problems.
* Fine-tuning Transformer models for specific downstream tasks.

## Technologies Used

* **Python**
* **PyTorch**: Deep Learning framework
* **NumPy**: Numerical computing
* **pandas**: Data manipulation and analysis
* **scikit-learn**: Utility functions (e.g., `LabelEncoder`)
* **Hugging Face Transformers** (likely used for `BertTokenizer` and potentially Transformer models)

## How to Run the Notebooks

To run these notebooks locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Leah0658/Deep-Learning-for-Sequential-Data.git]
    cd Deep-Learning-for-Sequential-Data
    ```
   

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)  # Or choose your CUDA version
    pip install numpy pandas scikit-learn transformers
    ```
    (You might need to adjust PyTorch installation command based on your system, e.g., for GPU support.)

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

    This will open Jupyter in your web browser, from where you can navigate and open each `.ipynb` file.

## Conclusion

These assignments provided a comprehensive hands-on experience in building, training, and evaluating deep learning models for sequential data. They highlight my ability to implement complex neural network architectures, preprocess diverse datasets, and apply advanced NLP techniques using industry-standard tools and frameworks.
````