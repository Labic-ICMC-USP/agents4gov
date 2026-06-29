import json
import numpy as np
from typing import List, Dict, Any
from pydantic import Field

class Tools:
    def __init__(self):
        pass

    def _compute_cosine_similarity(self, query_vec: List[float], dataset_vecs: List[List[float]]) -> np.ndarray:
        """
        Compute cosine similarity between query vector and multiple dataset vectors.

        Args:
            query_vec: Query vector
            dataset_vecs: List of dataset vectors

        Returns:
            Array of cosine similarity scores
        """
        from sklearn.metrics.pairwise import cosine_similarity

        query_array = np.array(query_vec).reshape(1, -1)
        dataset_array = np.array(dataset_vecs)

        similarities = cosine_similarity(query_array, dataset_array)
        return similarities[0]

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get text embeddings for a batch of texts using a local embedding model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Load model (you can cache this in __init__ for better performance)
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

            # Encode batch (much faster than encoding one by one)
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            return embeddings.tolist()

        except Exception as e:
            raise RuntimeError(
                f"No embedding service available. Please install sentence-transformers "
                f"(pip install sentence-transformers) Error: {str(e)}"
            )

    def _fetch_openml_datasets(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch datasets from OpenML API.

        Args:
            limit: Maximum number of datasets to fetch

        Returns:
            List of dataset dictionaries with metadata
        """
        try:
            import openml

            # List datasets with relevant metadata
            datasets_df = openml.datasets.list_datasets(output_format='dataframe')

            # Limit the number of datasets
            datasets_df = datasets_df.head(limit)

            # Convert to list of dictionaries
            datasets = []
            for idx, row in datasets_df.iterrows():
                dataset_dict = {
                    'did': int(row['did']),
                    'name': row.get('name', ''),
                    'description': row.get('description', ''),
                    'format': row.get('format', ''),
                    'uploader': row.get('uploader', ''),
                    'version': row.get('version', 1),
                    'status': row.get('status', ''),
                    'NumberOfInstances': row.get('NumberOfInstances', 0),
                    'NumberOfFeatures': row.get('NumberOfFeatures', 0),
                    'NumberOfClasses': row.get('NumberOfClasses', 0),
                    'NumberOfMissingValues': row.get('NumberOfMissingValues', 0),
                }
                datasets.append(dataset_dict)

            return datasets

        except ImportError:
            raise RuntimeError(
                "OpenML package not installed. Please install it with: pip install openml"
            )
        except Exception as e:
            raise RuntimeError(f"Error fetching OpenML datasets: {str(e)}")

    def _create_dataset_text(self, dataset: Dict[str, Any]) -> str:
        """
        Create a text representation of a dataset for embedding.

        Args:
            dataset: Dataset dictionary

        Returns:
            Text representation combining name and description
        """
        name = dataset.get('name', '')
        description = dataset.get('description', '')

        # Combine name and description for richer semantic matching
        text = f"{name}. {description}"

        # Truncate if too long (optional, depends on embedding model limits)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def search_openml_datasets(
        self,
        query: str = Field(
            ...,
            description="Natural language query to search for datasets (e.g., 'image classification datasets', 'medical diagnosis data', 'time series weather')"
        ),
        top_k: int = Field(
            default=5,
            description="Number of top similar datasets to return (default: 5)"
        ),
        max_datasets: int = Field(
            default=100,
            description="Maximum number of datasets to search through (default: 100)"
        )
    ) -> str:
        """
        Search for OpenML datasets using semantic similarity based on embeddings.

        This tool:
        1. Fetches datasets from OpenML
        2. Embeds the user query
        3. Embeds dataset names and descriptions
        4. Computes cosine similarity between query and datasets
        5. Returns top-k most similar datasets

        Args:
            query: Natural language search query
            top_k: Number of top results to return
            max_datasets: Maximum number of datasets to search

        Returns:
            JSON string with top-k most similar datasets and their metadata
        """

        try:
            # ========================================
            # STEP 1: FETCH DATASETS
            # ========================================

            datasets = self._fetch_openml_datasets(limit=max_datasets)

            if not datasets:
                return json.dumps({
                    'status': 'error',
                    'message': 'No datasets found in OpenML'
                }, ensure_ascii=False, indent=2)

            # ========================================
            # STEP 2: CREATE DATASET TEXTS
            # ========================================

            dataset_texts = [self._create_dataset_text(dataset) for dataset in datasets]

            # ========================================
            # STEP 3: EMBED QUERY AND DATASETS (BATCH)
            # ========================================

            # Combine query with dataset texts for batch embedding
            all_texts = [query] + dataset_texts
            all_embeddings = self._get_embeddings_batch(all_texts)

            # Extract query embedding and dataset embeddings
            query_embedding = all_embeddings[0]
            dataset_embeddings = all_embeddings[1:]

            # ========================================
            # STEP 4: COMPUTE SIMILARITIES
            # ========================================

            similarity_scores = self._compute_cosine_similarity(query_embedding, dataset_embeddings)

            similarities = [
                {
                    'dataset': dataset,
                    'similarity': float(score)
                }
                for dataset, score in zip(datasets, similarity_scores)
            ]

            # ========================================
            # STEP 5: SORT AND SELECT TOP-K
            # ========================================

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Select top-k
            top_results = similarities[:top_k]

            # ========================================
            # STEP 6: FORMAT RESULTS
            # ========================================

            results = []
            for item in top_results:
                dataset = item['dataset']
                similarity = item['similarity']

                results.append({
                    'dataset_id': dataset['did'],
                    'name': dataset['name'],
                    'description': dataset['description'][:200] + '...' if len(dataset.get('description', '')) > 200 else dataset.get('description', ''),
                    'similarity_score': round(similarity, 4),
                    'metadata': {
                        'num_instances': dataset.get('NumberOfInstances', 0),
                        'num_features': dataset.get('NumberOfFeatures', 0),
                        'num_classes': dataset.get('NumberOfClasses', 0),
                        'num_missing_values': dataset.get('NumberOfMissingValues', 0),
                        'format': dataset.get('format', ''),
                        'version': dataset.get('version', 1),
                        'uploader': dataset.get('uploader', ''),
                        'status': dataset.get('status', '')
                    },
                    'links': {
                        'openml_url': f"https://www.openml.org/d/{dataset['did']}",
                        'api_url': f"https://www.openml.org/api/v1/json/data/{dataset['did']}"
                    }
                })

            # ========================================
            # RETURN STRUCTURED RESPONSE
            # ========================================

            response = {
                'status': 'success',
                'query': query,
                'top_k': top_k,
                'total_searched': len(similarities),
                'results': results
            }

            return json.dumps(response, ensure_ascii=False, indent=2)

        # ========================================
        # ERROR HANDLING
        # ========================================

        except RuntimeError as e:
            # Handle specific runtime errors (missing dependencies, API issues)
            error_result = {
                'status': 'error',
                'error_type': 'runtime_error',
                'message': str(e),
                'query': query
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        except Exception as e:
            # Handle any other unexpected errors
            error_result = {
                'status': 'error',
                'error_type': 'unexpected_error',
                'message': f'Unexpected error during search: {str(e)}',
                'query': query
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
