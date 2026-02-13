import cv2
import dlib
import numpy as np
from skimage import transform as trans
from imutils import face_utils
import os

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    # The indices seem to be 1-based in some docs but 0-based in dlib access?
    # Dlib 68 point predictor:
    # 36-41: Left eye (so 37 is a point on left eye)
    # 42-47: Right eye (so 44 is a point on right eye)
    # 27-35: Nose (30 is nose tip)
    # 48-67: Mouth (49 is top lip left, 55 is top lip right/corner)
    
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
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

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        
        # NOTE: cropped_face is currently RGB because we passed 'rgb' to img_align_crop.
        # We will return it as RGB, suitable for PIL conversion.

        # Extract the all landmarks from the aligned face (optional, kept for compatibility if needed)
        # face_align = face_detector(cropped_face, 1)
        # if len(face_align) == 0:
        #     return None, None, None
        # landmark = predictor(cropped_face, face_align[0])
        # landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmarks, mask_face
    
    else:
        return None, None, None

class FaceExtractor:
    def __init__(self, predictor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found at {predictor_path}")
        self.face_predictor = dlib.shape_predictor(predictor_path)

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
                    face, landmarks, _ = extract_aligned_face_dlib(self.face_detector, self.face_predictor, frame)
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
            # extract_aligned_face_dlib expects BGR image
            face, landmarks, _ = extract_aligned_face_dlib(self.face_detector, self.face_predictor, frame)
            if face is not None:
                extracted_faces.append(face)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        
        return extracted_faces