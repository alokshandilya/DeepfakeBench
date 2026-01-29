import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Setup paths to ensure imports from the 'training' directory work correctly.
# The internal code of DeepfakeBench uses imports like 'from networks import ...' 
# which assumes 'training/' is in the python path.
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../DeepfakeBench/api
project_root = os.path.dirname(current_dir) # .../DeepfakeBench
training_dir = os.path.join(project_root, 'training')

if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from the local api module
try:
    from api.preprocessing_utils import FaceExtractor
except ImportError:
    # Fallback if running from within api directory directly
    from preprocessing_utils import FaceExtractor

# Import model components from the DeepfakeBench training codebase
# These imports rely on 'training_dir' being in sys.path
from detectors.xception_detector import XceptionDetector

class DeepfakeDetector:
    def __init__(self, model_weights_path, predictor_path, pretrained_imagenet_path, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")

        # Initialize FaceExtractor
        print("Initializing Face Extractor...")
        self.face_extractor = FaceExtractor(predictor_path)

        # Configuration for Xception
        # This mimics the structure expected by XceptionDetector.__init__
        self.config = {
            'model_name': 'xception',
            'backbone_name': 'xception',
            'backbone_config': {
                'mode': 'original',
                'num_classes': 2,
                'inc': 3,
                'dropout': False
            },
            'pretrained': pretrained_imagenet_path, # Path to ImageNet weights for backbone init
            'loss_func': 'cross_entropy', # Required by __init__ even if not used for inference
            'manualSeed': 1024
        }
        
        print("Initializing Xception Model...")
        # This will load the backbone with ImageNet weights first
        self.model = XceptionDetector(self.config)
        
        # Now load the specific Deepfake Detection trained weights
        print(f"Loading trained weights from {model_weights_path}...")
        try:
            ckpt = torch.load(model_weights_path, map_location=self.device)
            
            # Handle different checkpoint formats (full checkpoint vs state_dict)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
                
            # Remove 'module.' prefix if it exists (from DataParallel training)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=True)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            raise e

        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        # Based on training/config/detector/xception.yaml
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        self.transform = T.Compose([
            T.Resize((256, 256)), 
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
    def predict(self, video_path):
        """
        Run deepfake detection on a video file.
        
        Args:
            video_path (str): Path to the input video.
            
        Returns:
            dict: {
                'is_fake': bool,
                'fake_probability': float,
                'frames_processed': int,
                'error': str (optional)
            }
        """
        # 1. Extract faces
        # We process a subset of frames (e.g., 30) to save time
        faces = self.face_extractor.extract_faces(video_path, num_frames=30)
        
        if not faces:
            return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': 0,
                'error': 'No faces detected in the video.'
            }

        # 2. Preprocess faces
        processed_faces = []
        for face_img in faces:
            # face_img is RGB numpy array (H, W, 3)
            pil_img = Image.fromarray(face_img)
            tensor_img = self.transform(pil_img)
            processed_faces.append(tensor_img)
            
        # Stack into batch
        if not processed_faces:
             return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': 0,
                'error': 'Preprocessing failed.'
            }
            
        batch_input = torch.stack(processed_faces).to(self.device) # (N, 3, 256, 256)
        
        # 3. Inference
        try:
            with torch.no_grad():
                # XceptionDetector.forward expects a data_dict with 'image' key
                data_dict = {'image': batch_input}
                
                # forward(..., inference=True) returns:
                # {'cls': pred, 'prob': prob, 'feat': features}
                # prob is softmaxed probability for class 1 (Fake)
                output = self.model(data_dict, inference=True)
                probs = output['prob'] # Shape (N,)
                
            # 4. Aggregate results
            # Simple average of probabilities across all frames
            avg_prob = torch.mean(probs).item()
            
            # Threshold usually 0.5
            is_fake = avg_prob > 0.5
            
            return {
                'is_fake': is_fake,
                'fake_probability': avg_prob,
                'frames_processed': len(faces)
            }
        except Exception as e:
            return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': len(faces),
                'error': f"Inference failed: {str(e)}"
            }