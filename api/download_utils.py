import os
import urllib.request
import sys
from tqdm import tqdm

# URLs
PREDICTOR_URL = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat"
XCEPTION_WEIGHTS_URL = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth"
IMAGENET_XCEPTION_URL = "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "training/weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "xception_best.pth")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "training/pretrained")
PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, "xception-b5690688.pth")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading {url} to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add headers to avoid 403 Forbidden on some servers
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e

def check_and_download_dependencies():
    """
    Checks for necessary files and downloads them if missing.
    Returns the paths to the predictor, the model weights, and the pretrained weights.
    """
    print("Checking dependencies...")
    
    # 1. Dlib Shape Predictor
    download_file(PREDICTOR_URL, PREDICTOR_PATH)
    
    # 2. Xception Deepfake Detection Weights
    download_file(XCEPTION_WEIGHTS_URL, WEIGHTS_PATH)

    # 3. ImageNet Pretrained Weights (required for model initialization)
    download_file(IMAGENET_XCEPTION_URL, PRETRAINED_PATH)

    return PREDICTOR_PATH, WEIGHTS_PATH, PRETRAINED_PATH

if __name__ == "__main__":
    check_and_download_dependencies()