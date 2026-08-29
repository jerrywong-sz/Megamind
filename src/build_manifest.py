import os
import hashlib
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from pathlib import Path

def process_and_resave_image(img_path, output_dir, relative_path):
    """Checks for corruption, hashes, and forces EVERYTHING to standard JPEG."""
    try:
        with Image.open(img_path) as img:
            img.verify() # Corruption check
            
        with open(img_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Re-open for conversion
        with Image.open(img_path) as img:
            width, height = img.width, img.height
            img_rgb = img.convert('RGB') # Strip alpha channels
            
            # Save standardized JPEG to the new output directory
            save_path = Path(output_dir) / relative_path
            save_path.with_suffix('.jpg') # Force .jpg extension
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # This is the magic line that destroys format bias!
            img_rgb.save(save_path, format='JPEG', quality=95)
            
            return {"width": width, "height": height, "format": "JPEG", "hash": file_hash, "new_path": str(save_path)}
            
    except Exception:
        return None

def build_and_split_manifest(data_dir, dataset_name, generator_name, output_csv):
    records = []
    seen_hashes = set()
    
    print(f"Crawling {data_dir}...")
    for label_str in ["REAL", "FAKE"]:
        folder_path = Path(data_dir) / label_str
        label_val = 0.0 if label_str == "REAL" else 1.0 
        
        for img_path in folder_path.rglob("*.*"):
            if not img_path.is_file(): continue
            
            meta = get_image_meta(img_path)
            if meta is None or meta["hash"] in seen_hashes:
                continue
            seen_hashes.add(meta["hash"])
            
            # Convert /content/images/CIFAKE/train/REAL/img.jpg to train/REAL/img.jpg
            relative_path = str(img_path.relative_to(Path(data_dir).parent))

            records.append({
                "image_path": relative_path, "label": label_val, 
                "dataset": dataset_name, "generator": generator_name,
                "width": meta["width"], "height": meta["height"], "format": meta["format"]
            })

    df = pd.DataFrame(records)
    
    # Balance classes 50/50
    min_count = df['label'].value_counts().min()
    df_balanced = pd.concat([
        df[df['label'] == 0.0].sample(n=min_count, random_state=42),
        df[df['label'] == 1.0].sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 70/15/15 Split
    train, val, test = np.split(df_balanced, [int(.7*len(df_balanced)), int(.85*len(df_balanced))])
    train['split'], val['split'], test['split'] = 'train', 'val', 'test'
    
    final_manifest = pd.concat([train, val, test])
    final_manifest.to_csv(output_csv, index=False)
    print(f"Success! Manifest saved to {output_csv} with {len(final_manifest)} images.")

if __name__ == "__main__":
    build_and_split_manifest(
        data_dir="/content/images/CIFAKE/train", 
        dataset_name="CIFAKE", 
        generator_name="SD1.4", 
        output_csv="/content/images/cifake_manifest.csv"
    )