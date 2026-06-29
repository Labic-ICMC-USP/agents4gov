# OpenML Tools

This directory contains tools for working with OpenML datasets.

## Available Tools

### 1. OpenML Dataset Search (`openml_search.py`)

**Description:** Search for machine learning datasets on OpenML using semantic similarity based on embeddings. This tool enables natural language queries to find relevant datasets by computing similarity between the query and dataset descriptions. Uses batch embedding processing for efficiency.

**Main Method:** `search_openml_datasets(query: str, top_k: int = 5, max_datasets: int = 100) -> str`

### 2. OpenML Dataset Download (`openml_download.py`)

**Description:** Download datasets from OpenML by ID and automatically save as CSV. Returns the file path for immediate use with other tools.

**Main Method:** `download_openml_dataset(dataset_id: int, save_dir: str = "./datasets") -> str`

### 3. OpenML KNN Training (`openml_knn_train.py`)

**Description:** Train a K-Nearest Neighbors model with hyperparameter tuning using cross-validation. Automatically detects task type (classification/regression) and applies appropriate metrics and CV strategy.

**Main Method:** `train_knn_with_cv(data_path: str, target_column: str, n_neighbors_range: List[int] = [3, 5, 7, 9, 11], cv_folds: int = 5, ...) -> str`

---

## OpenML Dataset Search

### Features

- Natural language search queries for datasets
- Semantic similarity matching using embeddings
- Configurable number of results (top-k)
- Comprehensive dataset metadata retrieval
- Cosine similarity scoring between query and datasets
- Semantic search using sentence-transformers
- Returns structured JSON output with dataset details

### Parameters

- `query` (required): Natural language description of the desired dataset
  - Examples:
    - "image classification datasets"
    - "medical diagnosis data"
    - "time series weather data"
    - "text sentiment analysis"
- `top_k` (optional, default=5): Number of most similar datasets to return
- `max_datasets` (optional, default=100): Maximum number of datasets to search through

## Environment Variables

No environment variables required for embedding. Uses local sentence-transformers model `all-MiniLM-L6-v2`.


## Example Output

```json
{
  "status": "success",
  "query": "image classification datasets",
  "top_k": 5,
  "total_searched": 1000,
  "results": [
    {
      "dataset_id": 40927,
      "name": "mnist_784",
      "description": "The MNIST database of handwritten digits with 784 features. It is a subset of a larger set available from NIST...",
      "similarity_score": 0.8542,
      "metadata": {
        "num_instances": 70000,
        "num_features": 785,
        "num_classes": 10,
        "num_missing_values": 0,
        "format": "ARFF",
        "version": 1,
        "uploader": "Jan van Rijn",
        "status": "active"
      },
      "links": {
        "openml_url": "https://www.openml.org/d/40927",
        "api_url": "https://www.openml.org/api/v1/json/data/40927"
      }
    },
    {
      "dataset_id": 40996,
      "name": "Fashion-MNIST",
      "description": "Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples...",
      "similarity_score": 0.8213,
      "metadata": {
        "num_instances": 70000,
        "num_features": 785,
        "num_classes": 10,
        "num_missing_values": 0,
        "format": "ARFF",
        "version": 1,
        "uploader": "Joaquin Vanschoren",
        "status": "active"
      },
      "links": {
        "openml_url": "https://www.openml.org/d/40996",
        "api_url": "https://www.openml.org/api/v1/json/data/40996"
      }
    }
  ]
}
```

### Use Cases

- **Dataset Discovery**: Find datasets relevant to your research topic
- **Literature Review**: Identify datasets used in specific domains
- **Machine Learning Exploration**: Discover datasets for testing algorithms
- **Benchmarking**: Find standard datasets for model comparison
- **Education**: Locate datasets for teaching and learning

### How It Works

1. **Fetch Datasets**: Retrieves dataset metadata from OpenML API
2. **Batch Embedding**: Converts query and all dataset descriptions to vectors in a single batch (efficient)
3. **Similarity Computation**: Calculates cosine similarity using sklearn's optimized implementation
4. **Ranking**: Sorts datasets by similarity score
5. **Return Top-K**: Returns the most relevant datasets

### Technical Details

- Uses **sentence-transformers** with model `paraphrase-multilingual-mpnet-base-v2`
- **Batch processing** for embeddings (batch_size=32) for efficiency
- **Cosine similarity** computed via `sklearn.metrics.pairwise.cosine_similarity`
- All similarity scores normalized between -1 and 1

### Usage Example

```
Can you find datasets about medical diagnosis?
```

---

## OpenML Dataset Download

### Features

- Download datasets by OpenML ID
- Automatically saves as CSV format
- Comprehensive metadata extraction
- Feature information (categorical/numeric classification)
- File size reporting
- Returns absolute file path for chaining with other tools

### Parameters

- `dataset_id` (required): OpenML dataset ID (e.g., 40927 for MNIST)
- `save_dir` (optional, default="./datasets"): Directory to save the CSV file
  - Automatically creates directory if it doesn't exist
  - Filename format: `{dataset_name}_{dataset_id}.csv`

### Example Output

```json
{
  "status": "success",
  "dataset_id": 40927,
  "dataset_path": "/absolute/path/to/datasets/mnist_784_40927.csv",
  "metadata": {
    "dataset_id": 40927,
    "name": "mnist_784",
    "description": "The MNIST database of handwritten digits...",
    "version": 1,
    "format": "ARFF",
    "default_target_attribute": "class",
    "openml_url": "https://www.openml.org/d/40927",
    "num_features": 784,
    "num_instances": 70000
  },
  "data_info": {
    "saved_to_disk": true,
    "save_path": "/absolute/path/to/datasets/mnist_784_40927.csv",
    "file_size": "109.35 MB",
    "file_size_bytes": 114683392,
    "shape": {
      "features": [70000, 784],
      "target": [70000]
    },
    "feature_names": ["pixel_0_0", "pixel_0_1", "..."],
    "target_name": "class",
    "categorical_features": [],
    "numeric_features": ["pixel_0_0", "pixel_0_1", "..."]
  }
}
```

### Usage Example

```
Download dataset 40927
```

```
Download dataset 31 and save to ./my_datasets directory
```

---

## OpenML KNN Training

### Features

- **Automatic task detection**: Classifies as regression or classification based on target variable
- **Cross-validation**: Stratified K-Fold for classification, regular K-Fold for regression
- **Hyperparameter tuning**: Grid search over k-neighbors values
- **Multiple metrics**: Comprehensive evaluation metrics for both task types
- **Pipeline-based**: Includes StandardScaler for feature normalization
- **Model persistence**: Optionally save trained model with joblib

### Parameters

- `data_path` (required): Path to CSV dataset file
- `target_column` (required): Name of the target column
- `n_neighbors_range` (optional, default=[3, 5, 7, 9, 11]): List of k values to test
- `cv_folds` (optional, default=5): Number of cross-validation folds
- `random_state` (optional, default=42): Random seed for reproducibility
- `metric` (optional): Distance metric ('euclidean', 'manhattan', 'minkowski', 'chebyshev')
- `weights` (optional, default='uniform'): Weight function ('uniform' or 'distance')
- `save_model_path` (optional): Path to save the trained model

### Example Output (Classification)

```json
{
  "status": "success",
  "task_type": "classification",
  "dataset_info": {
    "data_path": "/path/to/dataset.csv",
    "total_samples": 1000,
    "num_features": 20,
    "target_column": "label",
    "num_classes": 3,
    "cv_folds": 5
  },
  "best_parameters": {
    "n_neighbors": 7,
    "weights": "uniform",
    "metric": "minkowski"
  },
  "hyperparameter_search": {
    "best_score": 0.9234,
    "all_params_scores": [
      {
        "params": {"n_neighbors": 3, "weights": "uniform"},
        "mean_score": 0.9123,
        "std_score": 0.0234
      },
      {
        "params": {"n_neighbors": 5, "weights": "uniform"},
        "mean_score": 0.9201,
        "std_score": 0.0198
      },
      {
        "params": {"n_neighbors": 7, "weights": "uniform"},
        "mean_score": 0.9234,
        "std_score": 0.0212
      }
    ]
  },
  "cross_validation_metrics": {
    "accuracy": {
      "mean": 0.9234,
      "std": 0.0212
    },
    "precision": {
      "mean": 0.9187,
      "std": 0.0223
    },
    "recall": {
      "mean": 0.9201,
      "std": 0.0198
    },
    "f1_score": {
      "mean": 0.9193,
      "std": 0.0205
    }
  },
  "model_info": {
    "saved": true,
    "save_path": "/path/to/model.joblib",
    "file_size_bytes": 1024
  }
}
```

### Example Output (Regression)

```json
{
  "status": "success",
  "task_type": "regression",
  "cross_validation_metrics": {
    "mse": {
      "mean": 12.45,
      "std": 2.34
    },
    "rmse": {
      "mean": 3.53,
      "std": 2.34
    },
    "mae": {
      "mean": 2.76,
      "std": 0.89
    },
    "r2_score": {
      "mean": 0.8765,
      "std": 0.0234
    }
  }
}
```

### Usage Example

```
Train a KNN model on the dataset at ./datasets/iris.csv with target column 'species'
```

```
Train KNN with k values [5, 10, 15] on ./data/housing.csv, target 'price', and save the model
```

### How It Works

1. **Load Data**: Reads CSV file into DataFrame
2. **Preprocessing**:
   - Handles missing values (mean for numeric, mode for categorical)
   - Encodes categorical features using LabelEncoder
   - Encodes target variable if classification
3. **Task Detection**: Automatically determines classification vs regression
4. **Cross-Validation Setup**: Creates stratified or regular K-Fold based on task
5. **Grid Search**: Tests all hyperparameter combinations using CV
6. **Metric Extraction**: Extracts all metrics from single GridSearchCV run
7. **Model Training**: Trains final model on all data with best parameters
8. **Save**: Optionally saves model pipeline and encoders

### Technical Details

- **Pipeline**: StandardScaler → KNN (ensures proper scaling in CV)
- **Single CV Run**: Uses GridSearchCV with multiple metrics (efficient)
- **Stratified CV**: Preserves class distribution in classification tasks
- **Feature Encoding**: Automatic handling of categorical variables
- **Model Package**: Saves pipeline, encoders, and metadata together

---

## General Information

### Dependencies

**All tools:**
```bash
pip install openml pandas numpy scikit-learn joblib sentence-transformers
```

**Breakdown:**
- `openml`: Dataset access
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning algorithms and metrics
- `joblib`: Model serialization
- `sentence-transformers`: Embeddings for search

### Environment Variables

No environment variables required. All tools use local libraries.

### Performance Considerations

**Search Tool:**
- **First Run**: May take longer due to model loading
- **Embedding Cache**: Consider caching embeddings for frequently searched datasets
- **Dataset Limit**: Adjust `max_datasets` parameter to balance speed vs. coverage

**Download Tool:**
- **Large Datasets**: May take time to download and save
- **Memory Usage**: Large datasets load into memory before saving
- **CSV Format**: Always saves as CSV for compatibility

**Training Tool:**
- **Cross-Validation**: Single GridSearchCV run computes all metrics efficiently
- **Memory Usage**: Entire dataset loaded into memory
- **Parallel Processing**: Uses `n_jobs=-1` for parallel CV
- **Large K Range**: More k values = longer training time

### Troubleshooting

#### Missing Dependencies

```bash
pip install openml pandas numpy scikit-learn joblib sentence-transformers
```

#### Slow Performance (Search)

- Reduce `max_datasets` parameter (default is now 100)
- Batch embedding is already optimized
- First run is slower due to model loading

#### Download Errors

- Verify dataset ID exists on OpenML
- Ensure sufficient disk space for large datasets
- Check write permissions for save_dir directory

#### Training Errors

- Verify target column name exists in dataset
- Check for sufficient samples (at least 2x cv_folds)
- Ensure dataset doesn't have all missing values
- For large datasets, reduce cv_folds or k range

## Additional Resources

- **[OpenML Website](https://www.openml.org/)** - Browse datasets online
- **[OpenML Python API Documentation](https://openml.github.io/openml-python/)** - Official API docs
- **[Sentence Transformers](https://www.sbert.net/)** - Embedding model documentation
- **[How to Create a Tool](../../docs/how_to_create_tool.md)** - Guide for creating your own tools
