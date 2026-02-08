# Text Classification with Deep Learning and Traditional Methods

This repository contains a comprehensive comparative study of text classification techniques, implementing and evaluating various machine learning approaches from traditional methods to modern deep learning architectures.

## Project Overview

This project implements and compares three distinct approaches to text classification:

1. **Deep Learning with LSTM Networks** - Advanced sequential modeling for text classification
2. **RNN with Multiple Embedding Strategies** - Comparative analysis of different word embedding techniques
3. **Traditional Machine Learning** - Classical approaches using word embeddings with logistic regression

The implementations provide a thorough comparison of performance, computational efficiency, and practical applicability across different text classification scenarios.

## Project Structure

```
├── LSTM_Text_Classification.ipynb                                    # LSTM implementation and experiments
├── RNN_Text_Classification_Embeddings.ipynb                         # RNN with various embedding strategies
├── Word_embedding_with_Traditional_Model_Logistic_Regression.ipynb   # Traditional ML approaches
└── README.md                                                         # Project documentation
```

## Implementation Details

### 1. LSTM Text Classification
- **Architecture**: Long Short-Term Memory networks for sequential text processing
- **Features**: Advanced preprocessing, model architecture optimization, performance evaluation
- **Dataset**: Text classification benchmarks
- **Metrics**: Accuracy, precision, recall, F1-score with comprehensive evaluation

### 2. RNN with Embedding Techniques
- **Embedding Types**:
  - Trainable word embeddings (learned during training)
  - Word2Vec Skip-gram model
  - Word2Vec Continuous Bag of Words (CBOW)
- **Architecture**: Recurrent Neural Networks optimized for different embedding strategies
- **Comparison**: Performance analysis across embedding methods

### 3. Traditional Methods with Word Embeddings
- **Models**: Logistic regression with various feature extraction techniques
- **Embeddings**: Pre-trained and custom word embeddings
- **Features**: Classical text preprocessing, feature engineering, statistical analysis
- **Baseline**: Performance benchmarks for deep learning comparison

## Technologies Used

### Core Libraries
- **TensorFlow**: Deep learning framework for neural network implementation
- **Scikit-learn**: Traditional machine learning algorithms and evaluation metrics
- **Gensim**: Word embedding models (Word2Vec) and text processing
- **NumPy**: Numerical computations and array handling
- **Pandas**: Data manipulation and analysis

### Development Environment
- **Jupyter Notebooks**: Interactive development and experimentation
- **Python 3.x**: Primary programming language

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Mugisha-isaac/machine-learning-techniques-formative-2
cd machine-learning-techniques-formative-2
```

2. Install required packages:
```bash
pip install -U pip setuptools wheel Cython
pip install gensim>=4.3.2
pip install tensorflow>=2.10.0
pip install scikit-learn>=1.0.0
pip install pandas numpy matplotlib seaborn
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

### Running the Experiments

1. **LSTM Text Classification**:
   - Open [LSTM_Text_Classification.ipynb](LSTM_Text_Classification.ipynb)
   - Execute cells sequentially for complete analysis
   - Review model architecture, training process, and results

2. **RNN with Different Embeddings**:
   - Open [RNN_Text_Classification_Embeddings.ipynb](RNN_Text_Classification_Embeddings.ipynb)
   - Compare performance across different embedding strategies
   - Analyze embedding quality and classification accuracy

3. **Traditional Methods**:
   - Open [Word_embedding_with_Traditional_Model_Logistic_Regression.ipynb](Word_embedding_with_Traditional_Model_Logistic_Regression.ipynb)
   - Execute baseline experiments
   - Compare with deep learning approaches

### Experiment Workflow

Each notebook follows a structured approach:
- Data preprocessing and exploration
- Model architecture definition
- Training and validation procedures
- Performance evaluation and visualization
- Comparative analysis and insights

## Key Features

- **Comprehensive Comparison**: Side-by-side evaluation of traditional and deep learning methods
- **Multiple Embedding Strategies**: Analysis of different word representation techniques
- **Performance Metrics**: Detailed evaluation using standard classification metrics
- **Visualization**: Graphical analysis of results and model behavior
- **Reproducible Research**: Clear documentation and consistent experimental setup

## Results and Analysis

The project provides insights into:
- Performance trade-offs between model complexity and accuracy
- Computational efficiency across different approaches
- Embedding quality impact on classification performance
- Practical considerations for real-world deployment

## Academic Context

This work represents a systematic study of text classification methodologies, suitable for:
- Machine learning course assignments
- Research into text processing techniques
- Comparative analysis of classical vs. modern approaches
- Educational demonstration of deep learning concepts

## Contributing

This repository serves as an educational resource. For improvements or extensions:
1. Ensure reproducibility of experiments
2. Maintain consistent documentation standards
3. Include proper evaluation metrics
4. Follow established coding conventions

## License

This project is developed for academic purposes. Please ensure proper attribution when using or referencing this work.

---

*Developed as part of Machine Learning Techniques coursework - Group 21*
