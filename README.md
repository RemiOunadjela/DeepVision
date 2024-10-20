# DeepVision: GPU-Accelerated Reverse Image Search Engine

## Overview

DeepVision is a high-performance reverse image search system that leverages deep learning and efficient similarity search techniques. This project enables users to quickly find visually similar images within large datasets using an image as a query, making it ideal for applications in digital asset management, content moderation, and visual data exploration.

### Key Features

- Feature extraction using a pre-trained ResNet50 model
- Efficient similarity search using FAISS (Facebook AI Similarity Search)
- Support for large image datasets
- Utilizes GPU acceleration (if available) for improved performance

## How It Works

1. **Feature Extraction**: The system uses a pre-trained ResNet50 model to extract high-dimensional feature vectors from images. These features capture essential visual characteristics of each image.

2. **Indexing**: FAISS is used to create an efficient index of these feature vectors, allowing for fast similarity searches even with large datasets.

3. **Similarity Search**: When a query image is provided, the system extracts its features and uses the FAISS index to find the most similar images in the dataset.

## Code Structure

- `main.py`: The main script that ties everything together.

Key functions:

- `load_model()`: Loads the pre-trained ResNet50 model.
- `get_transform()`: Defines image transformations for preprocessing.
- `extract_features_batch()`: Extracts features from a batch of images.
- `build_index()`: Creates a FAISS index from the extracted features.
- `search()`: Performs the similarity search for a given query image.
- `initialize_reverse_image_search()`: Sets up the search system, including loading or creating the index.
- `main()`: The entry point of the script, handling the overall flow.

## Usage

1. Place your database images in the `bulk_images` directory.
2. Place your query image(s) in the `that_one_image` directory.
3. Run the script: python main.py

## Code details
The script will process all images in `bulk_images`, build an index (or load an existing one), and then perform a search for each image in `that_one_image`.

### Requirements

- Python 3.x
- PyTorch
- torchvision
- FAISS
- PIL (Python Imaging Library)
- numpy
- tqdm

### Notes

- The first run will take longer as it needs to process all images and build the index.
- Subsequent runs will be faster as they can load the pre-computed index.
- Adding new images to the `bulk_images` directory will require rebuilding the index.

### Performance Considerations

- GPU acceleration is used if available, significantly speeding up feature extraction and search operations.
- For very large datasets, consider using SSD storage for faster image loading.
- The script uses batch processing to balance memory usage and performance.

### Future Improvements

- Implement incremental updating of the index for new images.
- Add support for different model architectures.
- Implement a user interface for easier interaction with the system.
