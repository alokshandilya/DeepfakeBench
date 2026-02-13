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
from detectors.effort_detector import EffortDetector

class DeepfakeDetector:
    def __init__(self, model_weights_path, predictor_path, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")

        # Initialize FaceExtractor
        print("Initializing Face Extractor...")
        self.face_extractor = FaceExtractor(predictor_path)

        # Configuration for Effort
        self.config = {
            'model_name': 'effort',
            'backbone_name': 'vit',
            'backbone_config': {
                'mode': 'original',
                'num_classes': 2,
                'inc': 3,
                'dropout': False
            },
            # 'pretrained': '...' # Not used by EffortDetector in the same way, handled internally or via transformers
        }
        
        print("Initializing Effort Model (this may download the base CLIP model from HuggingFace)...")
        self.model = EffortDetector(self.config)
        
        # Now load the specific Deepfake Detection trained weights
        print(f"Loading trained weights from {model_weights_path}...")
        try:
            ckpt = torch.load(model_weights_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                 state_dict = ckpt['model']
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
        # For Effort (based on CLIP), we typically use:
        # Mean: [0.48145466, 0.4578275, 0.40821073]
        # Std: [0.26862954, 0.26130258, 0.27577711]
        # Resolution: 224x224
        
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        self.transform = T.Compose([
            T.Resize((224, 224)), 
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
        # We process a subset of frames (e.g., 5) to save time
        faces = self.face_extractor.extract_faces(video_path, num_frames=5)
        
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
            
        batch_input = torch.stack(processed_faces).to(self.device) # (N, 3, 224, 224)
        
        # 3. Inference
        try:
            with torch.no_grad():
                # Forward expects a data_dict with 'image' key
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
                'frames_processed': len(faces),
                'model_used': 'Effort (ICML 2025 Spotlight)'
            }
        except Exception as e:
            return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': len(faces),
                'error': f"Inference failed: {str(e)}"
            }

    def predict_image(self, image_path):
        """
        Run deepfake detection on an image file.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            dict: {
                'is_fake': bool,
                'fake_probability': float,
                'frames_processed': int,
                'error': str (optional)
            }
        """
        # 1. Extract faces
        faces = self.face_extractor.extract_faces_from_image(image_path)
        
        if not faces:
            return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': 0,
                'error': 'No faces detected in the image.'
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
            
        batch_input = torch.stack(processed_faces).to(self.device) # (N, 3, 224, 224)
        
        # 3. Inference
        try:
            with torch.no_grad():
                # Forward expects a data_dict with 'image' key
                data_dict = {'image': batch_input}
                
                # forward(..., inference=True) returns:
                # {'cls': pred, 'prob': prob, 'feat': features}
                # prob is softmaxed probability for class 1 (Fake)
                output = self.model(data_dict, inference=True)
                probs = output['prob'] # Shape (N,)
                
            # 4. Aggregate results
            # Simple average of probabilities across all faces (usually just 1 for image)
            avg_prob = torch.mean(probs).item()
            
            # Threshold usually 0.5
            is_fake = avg_prob > 0.5
            
            return {
                'is_fake': is_fake,
                'fake_probability': avg_prob,
                'frames_processed': len(faces),
                'model_used': 'Effort (ICML 2025 Spotlight)'
            }
        except Exception as e:
            return {
                'is_fake': False,
                'fake_probability': 0.0,
                'frames_processed': len(faces),
                'error': f"Inference failed: {str(e)}"
            }