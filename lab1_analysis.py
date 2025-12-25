import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import random
import math

# --- Configuration ---
SIZES = [30000, 75000, 250000]
MISSING_PERCENTAGES = [0.03, 0.05, 0.10, 0.20, 0.30]
GENERATOR_SRC = "src/dataset_gen.cpp"
GENERATOR_EXE = "./a.out"
OUTPUT_FILENAME = "spb_purchases_dataset.csv"

# --- Helper Functions ---

def compile_generator():
    print("Compiling generator...")
    subprocess.run(["g++", GENERATOR_SRC, "-o", "a.out"], check=True)

def generate_dataset(size):
    print(f"Generating dataset with size {size}...")
    # Input: MasterCard Weight, Mir Weight, Sber Weight, TBank Weight, VTB Weight, Total Rows
    input_str = f"20 20\n20 20 20\n{size}\n"
    subprocess.run([GENERATOR_EXE], input=input_str, text=True, check=True)
    new_filename = f"dataset_{size}.csv"
    os.rename(OUTPUT_FILENAME, new_filename)
    return new_filename

# --- Helper Functions ---

def load_mappings():
    mappings = {}
    
    # 1. Stores
    try:
        stores_df = pd.read_csv("src/assets/stores.csv")
        # store_name is the first column
        stores = stores_df['store_name'].dropna().unique()
        # Start enumeration from 1 to avoid division by zero in relative error calculation
        mappings['store_name'] = {name: i+1 for i, name in enumerate(stores)}
        print(f"Loaded {len(stores)} stores from assets.")
    except Exception as e:
        print(f"Could not load stores from assets: {e}")

    # 2. Categories
    try:
        # category.csv has no header, format: name, brands_str, price, class
        # We need to handle the quoted brands string carefully.
        # Using python engine for more robust parsing
        cat_df = pd.read_csv("src/assets/category.csv", header=None, names=['name', 'brands', 'price', 'class'], quotechar='"')
        categories = cat_df['name'].dropna().unique()
        mappings['categories'] = {name: i+1 for i, name in enumerate(categories)}
        print(f"Loaded {len(categories)} categories from assets.")
    except Exception as e:
        print(f"Could not load categories from assets: {e}")

    # 3. Brands
    try:
        # brands_country.csv: brand, country
        brands_df = pd.read_csv("src/assets/brands_country.csv", header=None, names=['brand', 'country'])
        brands = brands_df['brand'].dropna().unique()
        mappings['brands'] = {name: i+1 for i, name in enumerate(brands)}
        print(f"Loaded {len(brands)} brands from assets.")
    except Exception as e:
        print(f"Could not load brands from assets: {e}")
        
    return mappings

def load_and_preprocess(filename):
    df = pd.read_csv(filename)
    mappings = load_mappings()
    
    # Apply asset-based mappings
    for col, mapping in mappings.items():
        if col in df.columns:
            # Map values, fill unknown with -1 or len(mapping)
            # Using map, then fillna
            df[col] = df[col].map(mapping)
            # If there are values in DF not in assets, we need to handle them.
            # Let's assign them new numbers.
            if df[col].isnull().any():
                missing_mask = df[col].isnull()
                unique_missing = df.loc[missing_mask, col].unique() # This will be NaNs if we mapped them?
                # Wait, map returns NaN for missing keys.
                # We need to know what the original values were to assign consistent IDs.
                # So we should probably combine asset mapping with dynamic mapping.
                
                # Reload original column to get unmapped values
                original_col = pd.read_csv(filename, usecols=[col])[col]
                unmapped = original_col[df[col].isnull()].unique()
                
                start_id = len(mapping)
                for i, val in enumerate(unmapped):
                    mapping[val] = start_id + i
                
                # Re-map
                df[col] = original_col.map(mapping)

    # Encode remaining categorical columns (including those we just mapped if they were not in assets?)
    # No, we handled them. Now handle others: datetime, coordinates, card_number, receipt_number
    
    # datetime: convert to timestamp
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime']).astype('int64') // 10**9
        
    # coordinates: map unique
    # card_number: map unique
    # receipt_number: map unique
    
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Add 1 to avoid 0s
        df[col] = le.fit_transform(df[col].astype(str)) + 1
        
    return df, mappings

def calculate_stats(df, name):
    print(f"\n--- Statistics for {name} ---")
    stats_df = pd.DataFrame()
    stats_df['Mean'] = df.mean()
    stats_df['Median'] = df.median()
    # Mode can be multiple, take the first one
    stats_df['Mode'] = df.mode().iloc[0]
    print(stats_df)
    return stats_df

def plot_distributions(df, name):
    print(f"Plotting distributions for {name}...")
    num_cols = len(df.columns)
    fig, axes = plt.subplots(math.ceil(num_cols / 3), 3, figsize=(15, 5 * math.ceil(num_cols / 3)))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.savefig(f"dist_{name}.png")
    plt.close()

def save_advanced_visualizations(df, labels, mappings, title, filename_prefix):
    """
    Generates multiple visualizations for clusters and saves them to 'claster_visualization' folder.
    """
    output_dir = "claster_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_vis = df.copy()
    df_vis['Cluster'] = labels
    
    print(f"Generating visualizations for {title} in {output_dir}...")
    
    # 1. PCA Plot
    numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.drop('Cluster')
    if len(numeric_cols) >= 2:
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(df_vis[numeric_cols])
        df_vis['PCA1'] = pca_res[:, 0]
        df_vis['PCA2'] = pca_res[:, 1]
        
        plt.figure(figsize=(10, 8))
        unique_labels = len(set(labels))
        palette = sns.color_palette("husl", unique_labels) if unique_labels > 10 else "tab10"
        sns.scatterplot(data=df_vis, x='PCA1', y='PCA2', hue='Cluster', palette=palette, s=50)
        plt.title(f'{title} - PCA')
        if unique_labels > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename_prefix}_PCA.png")
        plt.close()
        
    # 2. Top Categories
    if 'categories' in df_vis.columns and 'categories' in mappings:
        top_clusters = df_vis['Cluster'].value_counts().head(4).index
        cat_map = {v: k for k, v in mappings['categories'].items()}
        
        n_plots = len(top_clusters)
        if n_plots > 0:
            cols = 2 if n_plots > 1 else 1
            rows = math.ceil(n_plots / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
            
            # Ensure axes is always a flat list/array of axes
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, clust in enumerate(top_clusters):
                if i >= len(axes): break
                clust_data = df_vis[df_vis['Cluster'] == clust]
                counts = clust_data['categories'].value_counts().head(5)
                labels_str = [cat_map.get(int(idx), f"Unknown({idx})") for idx in counts.index]
                
                sns.barplot(x=counts.values, y=labels_str, ax=axes[i], palette='viridis')
                axes[i].set_title(f'Cluster {clust} - Top Categories')
                axes[i].set_xlabel('Count')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{filename_prefix}_Categories.png")
            plt.close()

def remove_chunks(df, percentage):
    """
    Removes data in chunks (matrices) to simulate outliers/block missingness.
    Optimized for speed using numpy.
    """
    # Ensure we work with floats to support NaN
    df_missing = df.copy().astype(float)
    data = df_missing.values
    n_rows, n_cols = data.shape
    total_cells = n_rows * n_cols
    cells_to_remove = int(total_cells * percentage)
    
    removed_count = 0
    
    # Batch size for random generation
    batch_size = 10000
    
    while removed_count < cells_to_remove:
        # Generate a batch of random chunks
        # Use slightly smaller range for start indices to ensure chunks fit
        r_h = np.random.randint(1, 6, size=batch_size)
        r_w = np.random.randint(1, 6, size=batch_size)
        r_r = np.random.randint(0, max(1, n_rows - 6), size=batch_size)
        r_c = np.random.randint(0, max(1, n_cols - 6), size=batch_size)
        
        for i in range(batch_size):
            if removed_count >= cells_to_remove:
                break
            
            h, w, r, c = r_h[i], r_w[i], r_r[i], r_c[i]
            
            # Slice view
            chunk = data[r:r+h, c:c+w]
            
            # Count non-NaNs to update counter accurately
            # (np.isnan is fast on small arrays)
            nans_in_chunk = np.isnan(chunk).sum()
            to_remove = (h * w) - nans_in_chunk
            
            if to_remove > 0:
                data[r:r+h, c:c+w] = np.nan
                removed_count += to_remove
                
    print(f"Chunks removed. Target: {cells_to_remove}, Actual: {removed_count}")
    
    # Re-assign modified data to dataframe
    df_missing[:] = data
    return df_missing

# --- Imputation Methods ---

def hot_deck_impute(df):
    print("Starting Hot-Deck...")
    df_imputed = df.copy()
    for col in df.columns:
        missing_mask = df_imputed[col].isna()
        if missing_mask.sum() > 0:
            observed_values = df_imputed.loc[~missing_mask, col]
            df_imputed.loc[missing_mask, col] = np.random.choice(observed_values, size=missing_mask.sum())
    print("Hot-Deck done.")
    return df_imputed

def spline_impute(df):
    print("Starting Spline...")
    df_imputed = df.copy()
    
    # Use 'cubic' interpolation which is faster and exact for restoration
    # instead of 'spline' which does smoothing and is slow for large N.
    # 'cubic' requires at least 4 points.
    
    for col in df.columns:
        # Skip if all NaN
        if df_imputed[col].isna().all():
            continue
            
        try:
            # 'cubic' is equivalent to spline of order 3 but uses different solver
            df_imputed[col] = df_imputed[col].interpolate(method='cubic', limit_direction='both')
        except Exception as e:
            # Fallback to linear
            # print(f"Cubic failed for {col}: {e}. Using linear.")
            df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
             
    # Fill any remaining NaNs (edges)
    df_imputed = df_imputed.bfill().ffill()
    print("Spline done.")
    return df_imputed

# --- Clustering (Forel) ---

def pearson_distance(x, y):
    # Pearson correlation distance: 1 - corr(x, y)
    # But Forel usually works in Euclidean space. 
    # If metric is Pearson, we use it.
    if len(x) < 2: return 0
    r, _ = stats.pearsonr(x, y)
    if np.isnan(r): return 1.0 # Max distance if undefined
    return 1 - r

def align_clusters(ref_labels, new_labels):
    """
    Aligns new_labels to match ref_labels based on maximum overlap.
    Returns aligned labels.
    """
    mapping = {}
    unique_new = np.unique(new_labels)
    
    for u_new in unique_new:
        if u_new == -1: continue # Skip noise
        
        # Indices where new_labels == u_new
        mask = (new_labels == u_new)
        # Get corresponding ref_labels
        refs = ref_labels[mask]
        
        if len(refs) == 0: 
            mapping[u_new] = u_new # Keep as is if no overlap (shouldn't happen with same indices)
            continue
        
        # Find most frequent ref label
        # scipy.stats.mode returns ModeResult(mode=array([val]), count=array([cnt]))
        mode_res = stats.mode(refs, keepdims=True)
        mode_ref = mode_res.mode[0]
        
        mapping[u_new] = mode_ref
        
    # Apply mapping
    aligned = np.array([mapping.get(x, x) for x in new_labels])
    return aligned

def forel_clustering(data, radius):
    """
    FOREL clustering algorithm.
    Returns cluster labels for each point in data.
    """
    unclustered_mask = np.ones(len(data), dtype=bool)
    labels = np.full(len(data), -1, dtype=int)
    cluster_id = 0
    
    while np.any(unclustered_mask):
        # Pick a random unclustered point as initial center
        current_indices = np.where(unclustered_mask)[0]
        center = data[np.random.choice(current_indices)]
        
        while True:
            distances = np.linalg.norm(data - center, axis=1)
            neighbors_mask = (distances <= radius) & unclustered_mask
            
            if not np.any(neighbors_mask):
                break
                
            new_center = np.mean(data[neighbors_mask], axis=0)
            
            if np.linalg.norm(new_center - center) < 1e-4:
                # Converged
                labels[neighbors_mask] = cluster_id
                unclustered_mask[neighbors_mask] = False
                cluster_id += 1
                break
            center = new_center
            
    return labels

def characterize_clusters(df, labels, mappings):
    """
    Analyze and print characteristics of each cluster.
    """
    df_labeled = df.copy()
    df_labeled['Cluster'] = labels
    
    unique_clusters = np.sort(df_labeled['Cluster'].unique())
    
    print(f"\n--- Cluster Characterization (Total Clusters: {len(unique_clusters)}) ---")
    
    # Reverse mappings for lookup: {col: {int_val: 'string_val'}}
    reverse_mappings = {}
    for col, mapping in mappings.items():
        reverse_mappings[col] = {v: k for k, v in mapping.items()}
        
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue # Skip noise if any
        
        cluster_data = df_labeled[df_labeled['Cluster'] == cluster_id]
        size = len(cluster_data)
        print(f"\nCluster {cluster_id} (Size: {size}, {size/len(df)*100:.1f}%)")
        
        # Print top 5 most distinctive features (highest deviation from global mean)
        # Or just print all for now since we have few features
        for col in df.columns:
            if col == 'Cluster': continue
            
            mean_val = cluster_data[col].mean()
            
            # Check if column has a mapping (categorical)
            if col in reverse_mappings:
                # Find nearest integer for mean to represent "typical" category
                nearest_int = int(round(mean_val))
                mapped_val = reverse_mappings[col].get(nearest_int, "Unknown")
                print(f"  {col}: Mode ~ {mapped_val} (Index: {mean_val:.1f})")
            else:
                print(f"  {col}: Mean = {mean_val:.2f}")

from sklearn.metrics import silhouette_score

def evaluate_clustering_quality(data, labels):
    """
    Calculates the quality of clustering.
    """
    if len(set(labels)) < 2:
        return -1.0
    # Use a sample for speed if data is large
    if len(data) > 5000:
        indices = np.random.choice(len(data), 5000, replace=False)
        return silhouette_score(data[indices], labels[indices])
    return silhouette_score(data, labels)

def spa_feature_selection(df, target_clusters=None):
    """
    Random Search with Adaptation (SPA) for feature selection.
    Since we don't have ground truth classes for 'feature selection' in unsupervised context usually,
    we maximize the clustering quality (Silhouette).
    
    df: DataFrame
    """
    features = list(df.columns)
    best_features = features.copy()
    
    # Initial clustering on all features
    scaler = StandardScaler()
    data = scaler.fit_transform(df[features])
    # Use KMeans for speed in feature selection loop instead of Forel
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
    labels = kmeans.fit_predict(data)
    best_score = evaluate_clustering_quality(data, labels)
    
    print(f"Initial score with all features: {best_score:.4f}")
    
    # SPA Parameters
    iterations = 20
    
    current_features = features.copy()
    
    for i in range(iterations):
        # Mutate: Flip a feature (include/exclude)
        if len(current_features) <= 1:
            # Force add if too small
            feat_to_add = random.choice([f for f in features if f not in current_features])
            candidate_features = current_features + [feat_to_add]
        else:
            # Randomly remove or add
            if random.random() < 0.5 and len(current_features) < len(features):
                # Add
                feat_to_add = random.choice([f for f in features if f not in current_features])
                candidate_features = current_features + [feat_to_add]
            else:
                # Remove
                feat_to_remove = random.choice(current_features)
                candidate_features = [f for f in current_features if f != feat_to_remove]
        
        if not candidate_features: continue
            
        # Evaluate
        data = scaler.fit_transform(df[candidate_features])
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
        labels = kmeans.fit_predict(data)
        score = evaluate_clustering_quality(data, labels)
        
        # Adaptation: Accept if better
        if score > best_score:
            best_score = score
            best_features = candidate_features
            current_features = candidate_features
            print(f"Iter {i}: New best score {best_score:.4f} with {len(best_features)} features")
        else:
            # Maybe accept with probability (Simulated Annealing style) - simplified here
            pass
            
    return best_features

# --- Main Execution ---

def main():
    # compile_generator() # Already compiled
    
    if len(sys.argv) > 1:
        try:
            selected_size = int(sys.argv[1])
            sizes_to_run = [selected_size]
        except ValueError:
            print("Invalid size provided. Usage: python3 lab1_analysis.py <size>")
            return
    else:
        print("No size provided. Running for default sizes.")
        sizes_to_run = SIZES

    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
    results = {}
    
    for size in sizes_to_run:
        if not os.path.exists(f"dataset_{size}.csv"):
             generate_dataset(size)
        
        filename = f"dataset_{size}.csv"
        df, mappings = load_and_preprocess(filename)
        
        # Check for zeros and replace them to avoid infinite relative error
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zeros_count = (df[numeric_cols] == 0).sum().sum()
        if zeros_count > 0:
            print(f"Warning: Found {zeros_count} zero values in numeric columns. Replacing with 1e-6 for stability.")
            df[numeric_cols] = df[numeric_cols].replace(0, 1e-6)
        
        # 1. Basic Stats & Plots
        calculate_stats(df, f"Original_{size}")
        plot_distributions(df, f"Original_{size}")

        # --- Feature Selection & Optimal Radius Tuning (MOVED TO START) ---
        print(f"\nRunning Feature Selection (SPA) on Original_{size}...")
        best_feats = spa_feature_selection(df)
        print(f"Best features: {best_feats}")
        
        # Prepare data for clustering tuning
        scaler = StandardScaler()
        data_best = scaler.fit_transform(df[best_feats])
        
        # Fix sample indices for consistent comparison across all methods
        if len(data_best) > 2000:
             # Use a fixed seed for reproducibility of the sample
             np.random.seed(42) 
             sample_indices = np.random.choice(len(data_best), 2000, replace=False)
             data_best_sample = data_best[sample_indices]
             df_best_sample = df.iloc[sample_indices]
        else:
             sample_indices = np.arange(len(data_best))
             data_best_sample = data_best
             df_best_sample = df

        # Find optimal radius
        current_radius = 1.5
        labels_best = []
        n_clusters_best = 0
        
        print("Tuning FOREL radius...")
        for _ in range(10):
            labels_best = forel_clustering(data_best_sample, radius=current_radius)
            n_clusters_best = len(np.unique(labels_best))
            if n_clusters_best >= 3:
                print(f"  Radius {current_radius:.2f} -> {n_clusters_best} clusters (Accepted)")
                break
            print(f"  Radius {current_radius:.2f} -> {n_clusters_best} clusters (Too few, decreasing)")
            current_radius -= 0.2
            if current_radius <= 0.1: break
            
        optimal_radius = current_radius
        
        print(f"Clustering Original with best features (radius={optimal_radius:.2f}) produced {n_clusters_best} clusters.")
        characterize_clusters(df_best_sample, labels_best, mappings)
        save_advanced_visualizations(df_best_sample, labels_best, mappings, f"Clustering Best Features {size}", f"clusters_best_{size}")
        
        # Store original labels for alignment
        original_labels_sample = labels_best

        # 2. Missing Values & Imputation Loop
        for pct in MISSING_PERCENTAGES:
            print(f"\nProcessing {pct*100}% missing values for size {size}...")
            df_missing = remove_chunks(df, pct)
            
            # Save missing
            df_missing.to_csv(f"datasets/dataset_{size}_missing_{int(pct*100)}.csv", index=False)
            
            # Impute
            df_hot_deck = hot_deck_impute(df_missing)
            df_spline = spline_impute(df_missing)
            
            # Save restored
            df_hot_deck.to_csv(f"datasets/dataset_{size}_restored_HotDeck_{int(pct*100)}.csv", index=False)
            df_spline.to_csv(f"datasets/dataset_{size}_restored_Spline_{int(pct*100)}.csv", index=False)
            
            # Evaluate Imputation Error
            numeric_cols_all = df.select_dtypes(include=[np.number]).columns
            
            for name, df_imp in [("HotDeck", df_hot_deck), ("Spline", df_spline)]:
                diff = np.abs(df[numeric_cols_all] - df_imp[numeric_cols_all])
                true_values = np.abs(df[numeric_cols_all])
                mask = true_values > 1e-3
                rel_error = pd.DataFrame(np.nan, index=df.index, columns=numeric_cols_all)
                rel_error[mask] = diff[mask] / true_values[mask]
                mean_rel_error = rel_error.mean().mean()
                print(f"Method: {name}, Size: {size}, Missing: {pct*100}%, Mean Rel Error: {mean_rel_error:.4f}")
                
                # Compare Clustering on Restored Data
                # We do this for all percentages or just one? User didn't specify, but usually interesting to see.
                # Let's do it for 10% as before to save time, or maybe all if fast. 
                # Given the user wants to "connect" them, let's do it for 10% as a representative case.
                if pct == 0.10:
                    print(f"  Clustering {name} restored data (Aligned to Original)...")
                    
                    # 1. Prepare data using BEST FEATURES
                    scaler_imp = StandardScaler()
                    data_imp = scaler_imp.fit_transform(df_imp[best_feats])
                    
                    # 2. Use SAME SAMPLE INDICES
                    data_imp_sample = data_imp[sample_indices]
                    df_imp_sample = df_imp.iloc[sample_indices]
                    
                    # 3. Cluster using OPTIMAL RADIUS
                    labels_imp = forel_clustering(data_imp_sample, radius=optimal_radius)
                    n_clusters_imp = len(np.unique(labels_imp))
                    
                    # 4. Align Labels to Original
                    aligned_labels = align_clusters(original_labels_sample, labels_imp)
                    
                    print(f"  {name} produced {n_clusters_imp} clusters (Radius: {optimal_radius:.2f}).")
                    
                    # 5. Visualize
                    characterize_clusters(df_imp_sample, aligned_labels, mappings)
                    save_advanced_visualizations(df_imp_sample, aligned_labels, mappings, 
                                               f"Clustering {name} (Restored & Aligned)", 
                                               f"clusters_{name}_{size}_aligned")

if __name__ == "__main__":
    main()
