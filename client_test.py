import asyncio
import aiohttp
import os
import time
import argparse

"""
DeepfakeBench API Stress Test & Benchmark Script

This script is designed to max out the Vast.ai server's concurrent 
processing capabilities. It splits a large directory of images into 
optimal batch sizes and sends them simultaneously to the server.

Optimal Server Launch Command (on Vast.ai):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uvicorn api.app:app --host 0.0.0.0 --port 8001 --workers 4

Recommended Test Size: 
Place 200 - 500 images in the 'test_images' directory to get a true 
benchmark of the server's throughput and parallel face extraction.
"""

# Hardcoded for the Vast.ai server running on port 8001
API_URL = "http://127.0.0.1:8001/detect_images"

# Optimal settings for an RTX 3090 + 64 Core CPU
BATCH_SIZE = 32
CONCURRENT_REQUESTS = 4

def chunker(seq, size):
    """Yield successive chunks from seq."""
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

async def send_batch(session: aiohttp.ClientSession, image_paths: list, batch_idx: int):
    """
    Sends a single batch of images to the /detect_images endpoint.
    Uses aiohttp.FormData to handle the multipart/form-data payload.
    """
    start_time = time.time()
    
    # Prepare the multipart form data
    data = aiohttp.FormData()
    files_to_close = []
    
    for path in image_paths:
        try:
            f = open(path, 'rb')
            files_to_close.append(f)
            # The field name MUST be 'files' to match the FastAPI endpoint signature
            data.add_field('files', f, filename=os.path.basename(path))
        except Exception as e:
            print(f"Error opening file {path}: {e}")
            
    try:
        # Send the POST request
        async with session.post(API_URL, data=data) as response:
            if response.status == 200:
                result = await response.json()
                elapsed = time.time() - start_time
                print(f"Batch {batch_idx:03d} [{len(image_paths)} images] completed in {elapsed:.2f}s")
                return result
            else:
                error_text = await response.text()
                print(f"Batch {batch_idx:03d} Failed with status {response.status}: {error_text}")
                return []
    except Exception as e:
        print(f"Batch {batch_idx:03d} Request Exception: {e}")
        return []
    finally:
        # Ensure all file handles are closed regardless of success/failure
        for f in files_to_close:
            f.close()

async def process_all_images(image_directory: str):
    """
    Scans the directory, chunks the files, and uses a semaphore to 
    limit the number of concurrent in-flight requests.
    """
    # 1. Gather all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    all_image_paths = []
    
    for filename in os.listdir(image_directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            all_image_paths.append(os.path.join(image_directory, filename))
            
    total_images = len(all_image_paths)
    if total_images == 0:
        print(f"No valid images found in {image_directory}")
        return
        
    print(f"Found {total_images} images to process.")
    print(f"Using Batch Size: {BATCH_SIZE}")
    print(f"Concurrent Connections: {CONCURRENT_REQUESTS}")
    
    # 2. Split into chunks
    batches = list(chunker(all_image_paths, BATCH_SIZE))
    total_batches = len(batches)
    print(f"Total batches to process: {total_batches}")
    
    overall_start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Semaphore restricts how many async tasks can run this block simultaneously.
        # This keeps the 4 Uvicorn workers completely saturated with data without 
        # overwhelming the server's network queue.
        sem = asyncio.Semaphore(CONCURRENT_REQUESTS) 
        
        async def bounded_send(batch, idx):
            async with sem:
                return await send_batch(session, batch, idx)
                
        # 3. Create all tasks
        tasks = [bounded_send(batch, idx) for idx, batch in enumerate(batches)]
        
        # 4. Run them in parallel and wait for all to finish
        results_lists = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single 1D list of results
        final_results = [item for sublist in results_lists for item in sublist]
        
    overall_time = time.time() - overall_start_time
    
    # 5. Calculate and print metrics
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Images Processed : {len(final_results)} / {total_images}")
    print(f"Total Time Taken       : {overall_time:.2f} seconds")
    
    if total_images > 0:
        throughput = total_images / overall_time
        print(f"Throughput             : {throughput:.2f} images/second")
        
        # Count fakes
        fakes = sum(1 for r in final_results if r.get('is_fake', False))
        errors = sum(1 for r in final_results if 'error' in r)
        print(f"Detected as Fake       : {fakes}")
        print(f"Processing Errors      : {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepfakeBench API Load Tester")
    parser.add_argument("--dir", type=str, default="test_images", help="Directory containing images to test")
    parser.add_argument("--url", type=str, default=API_URL, help="Full URL to the /detect_images endpoint")
    
    args = parser.parse_args()
    API_URL = args.url
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        print("Please create it and add some test images.")
        exit(1)
        
    # Run the asyncio event loop
    asyncio.run(process_all_images(args.dir))