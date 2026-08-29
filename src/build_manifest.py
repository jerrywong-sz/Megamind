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

def build_and_split_manifest(data_dir, output_dir, dataset_name, generator_name, output_csv):
    records = []
    seen_hashes = set()
    
    print(f"Crawling {data_dir} and standardizing to {output_dir}...")
    
    # We now support TAMPERED (label 2.0)
    for label_str, label_val in [("REAL", 0.0), ("FAKE", 1.0), ("TAMPERED", 2.0)]:
        folder_path = Path(data_dir) / label_str
        if not folder_path.exists():
            continue
            
        for img_path in folder_path.rglob("*.*"):
            if not img_path.is_file(): continue
            
            relative_path = img_path.relative_to(Path(data_dir))
            
            meta = process_and_resave_image(img_path, output_dir, relative_path)
            if meta is None or meta["hash"] in seen_hashes:
                continue
            seen_hashes.add(meta["hash"])
            
            # Enforce the Tampered Holdout Rule (PDF Page 4)
            split_override = "bonus" if label_val == 2.0 else None

            records.append({
                "image_path": meta["new_path"], "label": label_val, 
                "dataset": dataset_name, "generator": generator_name,
                "width": meta["width"], "height": meta["height"], 
                "format": meta["format"], "split_override": split_override
            })

    df = pd.DataFrame(records)
    
    # Balance Binary Classes (Real vs Fake)
    df_binary = df[df['label'].isin([0.0, 1.0])]
    min_count = df_binary['label'].value_counts().min()
    
    df_balanced = pd.concat([
        df_binary[df_binary['label'] == 0.0].sample(n=min_count, random_state=42),
        df_binary[df_binary['label'] == 1.0].sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 70/15/15 for Binary
    train, val, test = np.split(df_balanced, [int(.7*len(df_balanced)), int(.85*len(df_balanced))])
    train['split'], val['split'], test['split'] = 'train', 'val', 'test'
    
    # Merge back the Tampered (Bonus) data
    df_tampered = df[df['label'] == 2.0].copy()
    if not df_tampered.empty:
        df_tampered['split'] = df_tampered['split_override']
    
    final_manifest = pd.concat([train, val, test, df_tampered]).drop(columns=['split_override'])
    final_manifest.to_csv(output_csv, index=False)
    print(f"Success! Manifest saved to {output_csv} with {len(final_manifest)} standardized images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Original downloaded images")
    parser.add_argument("--output_dir", required=True, help="Where to save standardized JPEGs")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    build_and_split_manifest(
        args.data_dir, args.output_dir, 
        args.dataset_name, args.generator, args.output_csv
    )