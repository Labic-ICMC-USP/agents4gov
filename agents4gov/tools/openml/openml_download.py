import os
import json
from typing import Optional
from pydantic import Field

class Tools:
    def __init__(self):
        pass

    def _format_bytes(self, bytes_size: int) -> str:
        """
        Format bytes to human-readable string.

        Args:
            bytes_size: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    def download_openml_dataset(
        self,
        dataset_id: int = Field(
            ...,
            description="The OpenML dataset ID to download (e.g., 40927 for MNIST)"
        ),
        save_dir: str = Field(
            default="./datasets",
            description="Directory to save the dataset CSV file (default: ./datasets)"
        )
    ) -> str:
        """
        Download a dataset from OpenML by its ID and save as CSV.

        This tool:
        1. Fetches dataset from OpenML
        2. Saves as CSV file with features (X) and target (y)
        3. Returns the saved file path and metadata

        Args:
            dataset_id: OpenML dataset ID
            save_dir: Directory to save the CSV file

        Returns:
            JSON string with saved file path and metadata
        """

        try:
            import openml
            import pandas as pd

            # ========================================
            # STEP 1: FETCH DATASET METADATA
            # ========================================

            try:
                dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'error_type': 'dataset_not_found',
                    'message': f'Dataset with ID {dataset_id} not found: {str(e)}',
                    'dataset_id': dataset_id
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)

            # ========================================
            # STEP 2: EXTRACT METADATA
            # ========================================

            metadata = {
                'dataset_id': dataset.dataset_id,
                'name': dataset.name,
                'description': dataset.description,
                'version': dataset.version,
                'format': dataset.format,
                'upload_date': dataset.upload_date,
                'default_target_attribute': dataset.default_target_attribute,
                'row_id_attribute': dataset.row_id_attribute,
                'ignore_attributes': dataset.ignore_attribute,
                'language': dataset.language,
                'licence': dataset.licence,
                'url': dataset.url,
                'openml_url': f"https://www.openml.org/d/{dataset_id}"
            }

            # Extract features information
            if hasattr(dataset, 'features'):
                features_info = []
                for feature_name, feature_data in dataset.features.items():
                    features_info.append({
                        'name': feature_name,
                        'data_type': feature_data.data_type,
                        'is_target': feature_data.name == dataset.default_target_attribute,
                        'is_ignore': feature_data.name in (dataset.ignore_attribute or []),
                        'is_row_identifier': feature_data.name == dataset.row_id_attribute,
                        'number_missing_values': feature_data.number_missing_values
                    })
                metadata['features'] = features_info
                metadata['num_features'] = len([f for f in features_info if not f['is_target'] and not f['is_ignore']])
                metadata['num_instances'] = dataset.qualities.get('NumberOfInstances', 'unknown')

            # Extract qualities (statistics)
            if hasattr(dataset, 'qualities') and dataset.qualities:
                qualities = {
                    'num_instances': dataset.qualities.get('NumberOfInstances'),
                    'num_features': dataset.qualities.get('NumberOfFeatures'),
                    'num_classes': dataset.qualities.get('NumberOfClasses'),
                    'num_missing_values': dataset.qualities.get('NumberOfMissingValues'),
                    'num_instances_with_missing_values': dataset.qualities.get('NumberOfInstancesWithMissingValues'),
                    'num_numeric_features': dataset.qualities.get('NumberOfNumericFeatures'),
                    'num_symbolic_features': dataset.qualities.get('NumberOfSymbolicFeatures')
                }
                metadata['qualities'] = qualities

            # ========================================
            # STEP 3: DOWNLOAD DATA
            # ========================================

            try:
                # Get the data
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=dataset.default_target_attribute,
                    dataset_format='dataframe'
                )

                # Convert to DataFrames if not already
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=attribute_names)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name=dataset.default_target_attribute)

                # Data information
                data_info = {
                    'shape': {
                        'features': list(X.shape),
                        'target': list(y.shape)
                    },
                    'feature_names': list(X.columns),
                    'target_name': dataset.default_target_attribute,
                    'categorical_features': [attr for attr, is_cat in zip(attribute_names, categorical_indicator) if is_cat],
                    'numeric_features': [attr for attr, is_cat in zip(attribute_names, categorical_indicator) if not is_cat]
                }

                # ========================================
                # STEP 4: SAVE TO DISK
                # ========================================

                try:
                    # Create directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)

                    # Create filename from dataset name
                    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in dataset.name)
                    filename = f"{safe_name}_{dataset_id}.csv"
                    save_path = os.path.join(save_dir, filename)

                    # Combine X and y into single dataframe
                    data_df = X.copy()
                    data_df[dataset.default_target_attribute] = y

                    data_df.to_csv(save_path, index=False)

                    # Get file size
                    file_size = os.path.getsize(save_path)

                    data_info['saved_to_disk'] = True
                    data_info['save_path'] = os.path.abspath(save_path)
                    data_info['file_size'] = self._format_bytes(file_size)
                    data_info['file_size_bytes'] = file_size

                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'error_type': 'save_error',
                        'message': f'Error saving dataset to disk: {str(e)}',
                        'dataset_id': dataset_id
                    }
                    return json.dumps(error_result, ensure_ascii=False, indent=2)

            except Exception as e:
                error_result = {
                    'status': 'error',
                    'error_type': 'data_download_error',
                    'message': f'Error downloading dataset data: {str(e)}',
                    'dataset_id': dataset_id
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)

            # ========================================
            # BUILD RESPONSE
            # ========================================

            result = {
                'status': 'success',
                'dataset_id': dataset_id,
                'dataset_path': data_info['save_path'],
                'metadata': metadata,
                'data_info': data_info
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        # ========================================
        # ERROR HANDLING
        # ========================================

        except ImportError as e:
            error_result = {
                'status': 'error',
                'error_type': 'missing_dependency',
                'message': f'Required package not installed: {str(e)}. Please install with: pip install openml pandas',
                'dataset_id': dataset_id
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_result = {
                'status': 'error',
                'error_type': 'unexpected_error',
                'message': f'Unexpected error: {str(e)}',
                'dataset_id': dataset_id
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
