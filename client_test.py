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

async def send_batch(session: aiohttp.ClientSession, urls: list, batch_idx: int):
    """
    Sends a single batch of image URLs to the /detect_images endpoint.
    Uses a JSON payload.
    """
    start_time = time.time()
    
    payload = {"urls": urls}
            
    try:
        # Send the POST request
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                elapsed = time.time() - start_time
                print(f"Batch {batch_idx:03d} [{len(urls)} URLs] completed in {elapsed:.2f}s")
                return result
            else:
                error_text = await response.text()
                print(f"Batch {batch_idx:03d} Failed with status {response.status}: {error_text}")
                return []
    except Exception as e:
        print(f"Batch {batch_idx:03d} Request Exception: {e}")
        return []

async def process_all_urls(urls_file: str):
    """
    Reads URLs from a file, chunks them, and uses a semaphore to 
    limit the number of concurrent in-flight requests.
    """
    # 1. Gather all URLs
    all_urls = []
    
    with open(urls_file, 'r') as f:
        for line in f:
            url = line.strip()
            if url:
                all_urls.append(url)
            
    total_urls = len(all_urls)
    if total_urls == 0:
        print(f"No valid URLs found in {urls_file}")
        return
        
    print(f"Found {total_urls} URLs to process.")
    print(f"Using Batch Size: {BATCH_SIZE}")
    print(f"Concurrent Connections: {CONCURRENT_REQUESTS}")
    
    # 2. Split into chunks
    batches = list(chunker(all_urls, BATCH_SIZE))
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
    print(f"Total URLs Processed   : {len(final_results)} / {total_urls}")
    print(f"Total Time Taken       : {overall_time:.2f} seconds")
    
    if total_urls > 0:
        throughput = total_urls / overall_time
        print(f"Throughput             : {throughput:.2f} URLs/second")
        
        # Count fakes
        fakes = sum(1 for r in final_results if r.get('is_fake', False))
        
        # Collect errors
        error_results = [r for r in final_results if 'error' in r]
        errors = len(error_results)
        
        print(f"Detected as Fake       : {fakes}")
        print(f"Processing Errors      : {errors}")
        
        if errors > 0:
            print("\n" + "-"*40)
            print("ERROR DETAILS:")
            print("-"*40)
            for err in error_results:
                filename = err.get('url') or err.get('filename', 'Unknown URL')
                error_msg = err.get('error', 'Unknown error')
                print(f"- {filename}: {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepfakeBench API Load Tester")
    parser.add_argument("--file", type=str, default="test_urls.txt", help="Text file containing image URLs (one per line)")
    parser.add_argument("--url", type=str, default=API_URL, help="Full URL to the /detect_images endpoint")
    
    args = parser.parse_args()
    API_URL = args.url
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        print("Please create it and add some test URLs (one per line).")
        exit(1)
        
    # Run the asyncio event loop
    asyncio.run(process_all_urls(args.file))