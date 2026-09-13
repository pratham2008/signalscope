# src/prepare_splits.py
import os
import shutil

def copy_subset(src_dir, dst_dir, limit=10000):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        print(f"Skipping: {src_dir} does not exist.")
        return 0
        
    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)][:limit]
    
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    
    copied = len(files)
    print(f"Copied {copied} files from {src_dir} to {dst_dir}")
    return copied

def run_split_preparation():
    # Train: 15,000 real / 15,000 fake
    copy_subset("data/raw/cifake/train/REAL", "data/train/real", limit=15000)
    copy_subset("data/raw/cifake/train/FAKE", "data/train/fake", limit=15000)

    # In-Domain Val: 3,000 real / 3,000 fake
    copy_subset("data/raw/cifake/test/REAL", "data/val/real", limit=3000)
    copy_subset("data/raw/cifake/test/FAKE", "data/val/fake", limit=3000)

    # Pseudo-unseen Generator: 1,500 real / 1,500 fake
    copy_subset("data/raw/genimage_vqdm/val/nature", "data/unseen_test/real", limit=1500)
    copy_subset("data/raw/genimage_vqdm/val/ai", "data/unseen_test/fake", limit=1500)

    print("Data partitioning check complete.")

if __name__ == "__main__":
    run_split_preparation()