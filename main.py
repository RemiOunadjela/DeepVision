import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
from tqdm import tqdm
import pickle

# Use MPS if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} for computation")

def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_features_batch(model, transform, image_paths, batch_size=32):
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [transform(Image.open(path).convert('RGB')) for path in batch_paths]
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            batch_features = model(batch_tensor).squeeze().cpu().numpy()
        
        if len(batch_features.shape) == 1:
            batch_features = batch_features.reshape(1, -1)
        
        features.append(batch_features)
    
    return np.vstack(features)

def build_index(features):
    d = features.shape[1]
    n_clusters = min(400, max(1, int(np.sqrt(len(features)))))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters)
    index.train(features)
    index.add(features)
    return index

def search(query_feature, index, image_paths, k=5):
    distances, indices = index.search(query_feature.reshape(1, -1), k)
    return [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]

def save_features_and_index(features, index, image_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'features.npy'), features)
    with open(os.path.join(save_dir, 'image_paths.pkl'), 'wb') as f:
        pickle.dump(image_paths, f)
    faiss.write_index(index, os.path.join(save_dir, 'faiss_index.bin'))

def load_features_and_index(save_dir):
    features = np.load(os.path.join(save_dir, 'features.npy'), mmap_mode='r')
    with open(os.path.join(save_dir, 'image_paths.pkl'), 'rb') as f:
        image_paths = pickle.load(f)
    index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
    return features, index, image_paths

def get_image_paths(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in image_extensions]

def initialize_reverse_image_search(image_paths, save_dir='saved_index'):
    if os.path.exists(save_dir) and os.path.isfile(os.path.join(save_dir, 'faiss_index.bin')):
        print("Loading saved index and features...")
        features, index, saved_image_paths = load_features_and_index(save_dir)
        if saved_image_paths == image_paths:
            print("Loaded successfully!")
            model = load_model()
            transform = get_transform()
            return model, transform, index, features
        else:
            print("Saved image paths don't match current image paths. Rebuilding index...")

    print("Building new index...")
    model = load_model()
    transform = get_transform()
    features = extract_features_batch(model, transform, image_paths)
    index = build_index(features)
    save_features_and_index(features, index, image_paths, save_dir)
    print("Index built and saved successfully!")
    return model, transform, index, features

def main():
    bulk_images_dir = 'bulk_images'
    query_image_dir = 'that_one_image'
    save_dir = 'saved_index'
    
    database_image_paths = get_image_paths(bulk_images_dir)
    query_image_paths = get_image_paths(query_image_dir)

    if not database_image_paths:
        print(f"No images found in {bulk_images_dir}. Please add some images and try again.")
        return
    if not query_image_paths:
        print(f"No images found in {query_image_dir}. Please add a query image and try again.")
        return

    model, transform, index, features = initialize_reverse_image_search(database_image_paths, save_dir)
    
    for query_path in query_image_paths:
        print(f"\nSearching for similar images to: {query_path}")
        img = Image.open(query_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_feature = model(img_tensor).squeeze().cpu().numpy()
        results = search(query_feature, index, database_image_paths)
        for path, similarity in results:
            print(f"Similar image: {path}, Similarity: {similarity}")

if __name__ == "__main__":
    main()