import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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
async def detect_deepfake(request: Request):
    """
    Endpoint to detect if a video is a deepfake.
    
    Args:
        request (Request): The request object containing either a JSON body with 'file_path' or a file upload.
        
    Returns:
        JSON object containing 'is_fake', 'fake_probability', and 'frames_processed'.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    content_type = request.headers.get("content-type", "")
    target_path = None
    temp_path = None
    should_cleanup = False

    try:
        if "application/json" in content_type:
            try:
                data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            
            if "file_path" in data:
                target_path = data["file_path"]
                if not os.path.exists(target_path):
                    raise HTTPException(status_code=400, detail=f"File not found at path: {target_path}")
                print(f"Processing video from path: {target_path}")
            else:
                 raise HTTPException(status_code=400, detail="JSON body must contain 'file_path'")

        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            
            if not file:
                raise HTTPException(status_code=400, detail="No file provided in form data")

            if isinstance(file, str) or not hasattr(file, "filename"):
                raise HTTPException(status_code=400, detail="Form field 'file' must be a file upload")

            # Create a temporary file to save the uploaded video
            suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
            
            # Using tempfile.NamedTemporaryFile with delete=False
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
                target_path = temp_path
                should_cleanup = True
                print(f"Processing uploaded video: {file.filename} (saved to {temp_path})")
            finally:
                temp_file.close()
        else:
             raise HTTPException(status_code=400, detail="Content-Type must be application/json or multipart/form-data")

        # Run prediction
        result = detector.predict(target_path)
        
        # If there was an internal error in prediction (like no faces found)
        if 'error' in result:
            # Log unexpected errors but return the result structure
            if result['error'] not in ['No faces detected in the video.', 'Preprocessing failed.']:
                print(f"Inference error: {result['error']}")
        
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Cleanup: remove the temporary file
        if should_cleanup and temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temp file: {temp_path}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)