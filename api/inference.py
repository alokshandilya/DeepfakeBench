import sys
import os
import torch
import concurrent.futures
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

    def predict_batch(self, items):
        """
        Run deepfake detection on a batch of image files or bytes.
        Optimized to run inference on all faces from all images in a single batch.
        
        Args:
            items (List[dict]): List of dicts, each with 'path' or 'bytes' and 'filename'.
            
        Returns:
            List[dict]: List of result dictionaries corresponding to input images.
        """
        results = [None] * len(items)
        all_processed_faces = []
        # Stores (image_index, face_count) to map back results
        image_face_map = [] 
        
        def process_image(idx, item):
            try:
                image_bytes = item.get('bytes')
                image_path = item.get('path')
                filename = item.get('filename', image_path)
                
                if image_bytes is not None:
                    faces = self.face_extractor.extract_faces_from_image(image_bytes=image_bytes)
                else:
                    if not os.path.exists(image_path):
                        return idx, {
                            'file_path': filename,
                            'is_fake': False,
                            'fake_probability': 0.0,
                            'frames_processed': 0,
                            'error': 'File not found.'
                        }, []
                    faces = self.face_extractor.extract_faces_from_image(image_path=image_path)
                
                if not faces:
                    return idx, {
                        'file_path': filename,
                        'is_fake': False,
                        'fake_probability': 0.0,
                        'frames_processed': 0,
                        'error': 'No faces detected.'
                    }, []

                current_image_faces = []
                for face_img in faces:
                    pil_img = Image.fromarray(face_img)
                    tensor_img = self.transform(pil_img)
                    current_image_faces.append(tensor_img)
                
                return idx, None, current_image_faces
                
            except Exception as e:
                filename = item.get('filename', item.get('path'))
                return idx, {
                    'file_path': filename,
                    'is_fake': False,
                    'fake_probability': 0.0,
                    'frames_processed': 0,
                    'error': str(e)
                }, []

        # 1. Extract and preprocess faces in parallel since MTCNN (PyTorch) is thread-safe
        # Limit max_workers to prevent libgomp thread exhaustion and CUDA context errors with multiple Uvicorn workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), 2)) as executor:
            futures = [executor.submit(process_image, idx, item) for idx, item in enumerate(items)]
            
            processed_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            processed_results.sort(key=lambda x: x[0])
            
            for idx, err_result, current_image_faces in processed_results:
                if err_result is not None:
                    results[idx] = err_result
                    image_face_map.append((idx, 0))
                else:
                    all_processed_faces.extend(current_image_faces)
                    image_face_map.append((idx, len(current_image_faces)))

        # If no faces found in any images
        if not all_processed_faces:
            # Fill remaining Nones with defaults if any logic slipped
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = {
                        'file_path': items[i].get('filename', items[i].get('path')),
                        'is_fake': False,
                        'fake_probability': 0.0,
                        'frames_processed': 0,
                        'error': 'No faces processed.'
                    }
            return results

        # 2. Batch Inference (GPU bound part)
        try:
            batch_input = torch.stack(all_processed_faces).to(self.device)
            
            with torch.no_grad():
                data_dict = {'image': batch_input}
                output = self.model(data_dict, inference=True)
                probs = output['prob'] # Shape (Total_Faces,)
                
            # 3. Distribute results
            current_face_ptr = 0
            for idx, face_count in image_face_map:
                if face_count == 0:
                    continue
                    
                # Slice the probabilities for this image
                img_probs = probs[current_face_ptr : current_face_ptr + face_count]
                current_face_ptr += face_count
                
                # Aggregate
                avg_prob = torch.mean(img_probs).item()
                is_fake = avg_prob > 0.5
                
                results[idx] = {
                    'file_path': items[idx].get('filename', items[idx].get('path')),
                    'is_fake': is_fake,
                    'fake_probability': avg_prob,
                    'frames_processed': face_count,
                    'model_used': 'Effort (ICML 2025 Spotlight)'
                }

        except Exception as e:
            # Fatal inference error affecting all remaining
            err_msg = f"Batch inference failed: {str(e)}"
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = {
                        'file_path': items[i].get('filename', items[i].get('path')),
                        'is_fake': False,
                        'fake_probability': 0.0,
                        'frames_processed': 0,
                        'error': err_msg
                    }
        
        return results