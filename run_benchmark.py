import os
import requests
import csv
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/detect"
BENCHMARK_DIR = "benchmark"
OUTPUT_CSV = "benchmark_results.csv"

def run_benchmark():
    # Ensure benchmark directory exists
    if not os.path.exists(BENCHMARK_DIR):
        print(f"Error: Benchmark directory '{BENCHMARK_DIR}' not found.")
        return

    # Prepare CSV data
    results = []
    
    # Get list of video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    files = [f for f in os.listdir(BENCHMARK_DIR) if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not files:
        print(f"No video files found in '{BENCHMARK_DIR}'.")
        return

    print(f"Found {len(files)} videos. Starting benchmark...")

    for filename in files:
        file_path = os.path.join(BENCHMARK_DIR, filename)
        print(f"Processing {filename}...", end=" ", flush=True)

        try:
            with open(file_path, 'rb') as f:
                # Prepare the multipart/form-data request
                # We specify the filename and content_type to ensure it matches typical browser/curl behavior
                files_payload = {'file': (filename, f, 'video/mp4')}
                
                response = requests.post(API_URL, files=files_payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract fields based on the sample response provided
                    # {
                    #   "is_fake": true,
                    #   "fake_probability": 0.8981175422668457,
                    #   "frames_processed": 5,
                    #   "model_used": "Effort (ICML 2025 Spotlight)",
                    #   "video_path": "/tmp/tmpi64ijm1t.mp4"
                    # }
                    
                    result_entry = {
                        "filename": filename,
                        "is_fake": data.get("is_fake"),
                        "fake_probability": data.get("fake_probability"),
                        "frames_processed": data.get("frames_processed"),
                        "model_used": data.get("model_used"),
                        "status": "Success"
                    }
                    
                    prob = result_entry['fake_probability']
                    prob_str = f"{prob:.4f}" if prob is not None else "N/A"
                    print(f"Done. Fake: {result_entry['is_fake']} ({prob_str})")
                    results.append(result_entry)
                else:
                    print(f"Failed. Status Code: {response.status_code}")
                    print(f"Response: {response.text}")
                    results.append({
                        "filename": filename,
                        "status": f"Failed ({response.status_code})"
                    })

        except requests.exceptions.ConnectionError:
            print("Failed. Could not connect to API. Is the server running?")
            results.append({
                "filename": filename,
                "status": "Connection Error"
            })
        except Exception as e:
            print(f"Failed. Error: {str(e)}")
            results.append({
                "filename": filename,
                "status": f"Error: {str(e)}"
            })

    # Write results to CSV
    if results:
        fieldnames = ["filename", "is_fake", "fake_probability", "frames_processed", "model_used", "status"]
        
        try:
            with open(OUTPUT_CSV, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    # Filter row to only include keys in fieldnames to avoid DictWriter errors
                    # if we added extra debug info or partial failures
                    filtered_row = {k: row.get(k) for k in fieldnames}
                    writer.writerow(filtered_row)
            print(f"\nBenchmark complete. Results saved to '{OUTPUT_CSV}'.")
        except IOError as e:
            print(f"\nError writing CSV file: {e}")

if __name__ == "__main__":
    run_benchmark()