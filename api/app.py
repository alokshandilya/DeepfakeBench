import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn

# Ensure we can import from local modules
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
# Depending on how this is run (python -m api.app vs python api/app.py), imports might vary slightly.
# We try relative imports first if running as a package, otherwise absolute.
try:
    from api.download_utils import check_and_download_dependencies
    from api.inference import DeepfakeDetector
except ImportError:
    from download_utils import check_and_download_dependencies
    from inference import DeepfakeDetector

# Global variable to hold the detector instance
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup and shutdown logic.
    Downloads dependencies and initializes the model on startup.
    """
    global detector
    print("Startup: Checking and downloading dependencies...")
    try:
        predictor_path, weights_path = check_and_download_dependencies()
        
        print("Startup: Initializing DeepfakeDetector (Effort Model)...")
        # Initialize the detector and store it in the app state
        detector = DeepfakeDetector(
            model_weights_path=weights_path, 
            predictor_path=predictor_path
        )
        print("Startup: Model initialized successfully.")
    except Exception as e:
        print(f"Startup Error: Failed to initialize model. {e}")
        # We raise here to prevent the app from starting in a broken state
        raise e
        
    yield
    
    # Shutdown logic (if any)
    print("Shutdown: Cleaning up resources...")
    detector = None

app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Deepfake Detection API is running. Use /detect to check a video."}

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Endpoint to detect if a video is a deepfake.
    
    Args:
        file (UploadFile): Video file to analyze.
        
    Returns:
        JSON object containing 'is_fake', 'fake_probability', and 'frames_processed'.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    # Create a temporary file to save the uploaded video
    # We need to save it to disk because cv2.VideoCapture reads from a file path
    # Using tempfile.NamedTemporaryFile with delete=False to ensure it persists for opencv to read
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        try:
            # Write uploaded content to the temp file
            shutil.copyfileobj(file.file, temp_video)
            temp_video_path = temp_video.name
            temp_video.close() # Close file handle so other processes can access it if needed

            print(f"Processing video: {file.filename} (saved to {temp_video_path})")
            
            # Run prediction
            result = detector.predict(temp_video_path)
            
            # If there was an internal error in prediction (like no faces found)
            if 'error' in result:
                # Log unexpected errors but return the result structure
                if result['error'] not in ['No faces detected in the video.', 'Preprocessing failed.']:
                    print(f"Inference error: {result['error']}")
            
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        finally:
            # Cleanup: remove the temporary file
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print(f"Cleaned up temp file: {temp_video_path}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)