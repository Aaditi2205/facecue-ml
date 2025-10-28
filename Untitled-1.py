class RealDatasetCollector:
    # ...existing code...

    def download_kaggle_datasets(self):
        """Download specific Kaggle datasets for health/lifestyle analysis"""
        kaggle_datasets = {
            "sleep-health-lifestyle": {
                "url": "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset",
                "description": "Sleep patterns, lifestyle habits, and health indicators",
                "kaggle_id": "uom190346a/sleep-health-and-lifestyle-dataset"
            },
            "nutrition-health-survey": {
                "url": "https://www.kaggle.com/datasets/nutrition-health-survey",
                "description": "Comprehensive nutrition and health survey data",
                "kaggle_id": "nutrition-health-survey"
            },
            "lifestyle-wellness": {
                "url": "https://www.kaggle.com/datasets/lifestyle-wellness-data",
                "description": "Daily lifestyle and wellness metrics",
                "kaggle_id": "lifestyle-wellness-data"
            },
            "medical-questionnaire": {
                "url": "https://www.kaggle.com/datasets/medical-questionnaire-responses",
                "description": "Medical questionnaire responses and health status",
                "kaggle_id": "medical-questionnaire-responses"
            },
            "diet-nutrition-analysis": {
                "url": "https://www.kaggle.com/datasets/diet-nutrition-analysis",
                "description": "Dietary patterns and nutritional analysis",
                "kaggle_id": "diet-nutrition-analysis"
            },
            # Facial image datasets on Kaggle (for later multimodal work)
            "utkface-kaggle": {
                "url": "https://www.kaggle.com/datasets/utkface/utkface",
                "description": "UTKFace mirror on Kaggle (if available)",
                "kaggle_id": "utkface/utkface"
            },
            "ffhq-kaggle": {
                "url": "https://www.kaggle.com/datasets/ffhq-dataset",
                "description": "FFHQ mirror on Kaggle (if available)",
                "kaggle_id": "ffhq-dataset"
            },
            "celeba": {
                "url": "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset",
                "description": "CelebA dataset (faces with attributes)",
                "kaggle_id": "jessicali9530/celeba-dataset"
            },
            "lfw": {
                "url": "https://www.kaggle.com/datasets/serengil/lfw",
                "description": "LFW (Labeled Faces in the Wild)",
                "kaggle_id": "serengil/lfw"
            }
        }

        print("=== Kaggle Datasets to Download ===")
        for name, info in kaggle_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            if 'kaggle_id' in info:
                print(f"  Command: kaggle datasets download -d {info['kaggle_id']} -p data/kaggle/")
        return kaggle_datasets

    # ...existing code...

    def download_facial_image_datasets(self):
        """Download facial image datasets for health analysis"""
        image_datasets = {
            "utkface": {
                "url": "https://susanqq.github.io/UTKFace/",
                "description": "20,000+ face images with age/gender/ethnicity labels",
                "size": "~2GB",
                "features": "Age, gender, ethnicity annotations",
                "notes": "Manual download or Kaggle mirror recommended"
            },
            "ffhq": {
                "url": "https://github.com/NVlabs/ffhq-dataset",
                "description": "High-quality diverse face images",
                "size": "~7GB",
                "features": "High resolution, diverse demographics"
            },
            "fitzpatrick17k": {
                "url": "https://github.com/sfu-mial/awesome-skin-image-analysis-datasets",
                "description": "Clinical images with skin type labels",
                "size": "~500MB",
                "features": "Skin type classification, clinical annotations"
            },
            "celeba": {
                "url": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
                "description": "Celebrity faces with attribute annotations",
                "size": "~1.3GB",
                "features": "40+ facial attributes"
            },
            "lfw": {
                "url": "http://vis-www.cs.umass.edu/lfw/",
                "description": "Labeled Faces in the Wild",
                "size": "~200MB",
                "features": "Face recognition dataset"
            }
        }

        print("\n=== Facial Image Datasets ===")
        for name, info in image_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            print(f"  Size: {info['size']}")
            print(f"  Features: {info['features']}")
            if 'notes' in info:
                print(f"  Notes: {info['notes']}")

        return image_datasets

    # ...existing code...