import cv2
import torch
import numpy as np
from skimage import transform as trans
import os
from facenet_pytorch import MTCNN

def extract_aligned_face_mtcnn(mtcnn, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with MTCNN
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        # Align and crop the face using the 5 landmarks natively provided by MTCNN
        # landmarks[0] is the landmarks for the largest face
        cropped_face, mask_face = img_align_crop(rgb, landmarks[0], outsize=(res, res), mask=mask)
        
        return cropped_face, landmarks[0], mask_face
    else:
        return None, None, None

class FaceExtractor:
    def __init__(self, predictor_path=None):
        # predictor_path is ignored now since we use MTCNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize MTCNN. select_largest=True ensures it returns the biggest face first.
        self.mtcnn = MTCNN(keep_all=True, select_largest=True, device=self.device)

    def extract_faces(self, video_path, num_frames=30):
        """
        Extract aligned faces from the video.
        Returns a list of numpy arrays (RGB images).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # Fallback for streams or unknown lengths
             frame_count = 100 
             
        # Uniform sampling
        if num_frames and frame_count > num_frames:
            indices = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int)
            indices = set(indices)
        else:
            indices = set(range(frame_count))

        extracted_faces = []
        
        # Loop through frames
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in indices:
                # extract_aligned_face_dlib expects BGR image (it converts to RGB internally), 
                # but we modified it above. Let's double check.
                # In extract_aligned_face_dlib: rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # So we pass BGR 'frame' directly.
                try:
                    face, landmarks, _ = extract_aligned_face_mtcnn(self.mtcnn, frame)
                    if face is not None:
                        extracted_faces.append(face)
                except Exception as e:
                    print(f"Error processing frame {current_frame}: {e}")
            
            current_frame += 1
            if current_frame > frame_count:
                break
        
        cap.release()
        return extracted_faces

    def extract_faces_from_image(self, image_path):
        """
        Extract aligned faces from a single image file.
        Returns a list of numpy arrays (RGB images).
        """
        # cv2.imread loads in BGR format
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error opening image file {image_path}")
            return []

        extracted_faces = []
        try:
            # extract_aligned_face_mtcnn expects BGR image (it converts internally)
            face, landmarks, _ = extract_aligned_face_mtcnn(self.mtcnn, frame)
            if face is not None:
                extracted_faces.append(face)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        
        return extracted_faces