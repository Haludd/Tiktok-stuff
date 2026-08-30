import os
import shutil
import zipfile

# If you want to hardcode your token directly in code so you never deal with .env issues:
# (Replace these with your actual Kaggle credentials if you want)
os.environ['KAGGLE_USERNAME'] = "Haludd"
os.environ['KAGGLE_KEY'] = "KGAT_b8588598d3ee8cbb46977e777854e7fa"

from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_organize_cifake():
    api = KaggleApi()
    api.authenticate()

    dataset_slug = "birdy654/cifake-real-and-ai-generated-synthetic-images"
    zip_name = "cifake-real-and-ai-generated-synthetic-images.zip"

    print("Downloading CIFAKE dataset from Kaggle...")
    api.dataset_download_files(dataset_slug, path=".", unzip=False)

    print("Extracting files...")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall("temp_data")

    print("Organizing folders into 'dataset_root/'...")
    
    # Kaggle CIFAKE layout mapping: temp_data/train/REAL -> dataset_root/train/real
    mapping = {
        ("train", "REAL"): ("train", "real"),
        ("train", "FAKE"): ("train", "fake"),
        ("test", "REAL"): ("val", "real"),       # Using test split as validation
        ("test", "FAKE"): ("val", "fake")
    }

    for (src_split, src_cls), (dest_split, dest_cls) in mapping.items():
        src_dir = os.path.join("temp_data", src_split, src_cls)
        dest_dir = os.path.join("dataset_root", dest_split, dest_cls)
        
        os.makedirs(dest_dir, exist_ok=True)
        
        if os.path.exists(src_dir):
            for file_name in os.listdir(src_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    shutil.move(
                        os.path.join(src_dir, file_name), 
                        os.path.join(dest_dir, file_name)
                    )

    # Cleanup temporary downloads
    if os.path.exists(zip_name):
        os.remove(zip_name)
    if os.path.exists("temp_data"):
        shutil.rmtree("temp_data")

    print("\nDataset successfully downloaded and organized via code!")
    print("Ready to run: python3 train.py")

if __name__ == "__main__":
    download_and_organize_cifake()