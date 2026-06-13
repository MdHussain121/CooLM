import os
import json
import webbrowser
import threading
import shutil
import urllib.request
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from tools.onnx_chat import CooONNXInference

def download_file(url, dest_path, description="File"):
    print(f"Downloading {description} from Hugging Face...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            start_time = time.time()
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        speed = downloaded / max((time.time() - start_time), 0.001) / (1024 * 1024)  # MB/s
                        print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB) - Speed: {speed:.2f} MB/s", end="", flush=True)
                    else:
                        print(f"\rDownloaded: {downloaded/(1024*1024):.1f}MB", end="", flush=True)
            print(f"\nSuccessfully downloaded {description} to {dest_path}!")
            return True
    except Exception as e:
        print(f"\nError downloading {description}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass
        return False

# Initialize the inference engine once, optimized for reuse.
print("Starting CooLM Local Web Server...")

print("========================================")
print(" CooLM Chat Server Initialization")
print("========================================")
print("Which model version would you like to use?")
print("1) Hugging Face version (download from MdHussain121/coolm-42M)")
print("2) Local version (your own trained/exported model)")
print("========================================")

choice = ""
while choice not in ("1", "2"):
    choice = input("Enter your choice (1 or 2): ").strip()

onnx_path = None
tokenizer_path = None

model_url = "https://huggingface.co/MdHussain121/coolm-42M/resolve/main/model.onnx"
tokenizer_url = "https://huggingface.co/MdHussain121/coolm-42M/resolve/main/tokenizer.json"

if choice == "1":
    # User wants Hugging Face version
    model_ok = os.path.exists("docs/model_hf.onnx")
    tokenizer_ok = os.path.exists("data/tokenizer_hf.json")
    
    if model_ok and tokenizer_ok:
        print("Using cached Hugging Face version.")
    else:
        print("Hugging Face model files not found locally.")
        if not model_ok:
            model_ok = download_file(model_url, "docs/model_hf.onnx", "CooLM ONNX model")
        if not tokenizer_ok:
            tokenizer_ok = download_file(tokenizer_url, "data/tokenizer_hf.json", "CooLM Tokenizer")
            
    if model_ok and tokenizer_ok:
        onnx_path = "docs/model_hf.onnx"
        tokenizer_path = "data/tokenizer_hf.json"
    else:
        print("\nError: Failed to obtain Hugging Face model/tokenizer files.")
        if os.path.exists("docs/model.onnx") and os.path.exists("data/tokenizer.json"):
            print("Falling back to local version...")
            onnx_path = "docs/model.onnx"
            tokenizer_path = "data/tokenizer.json"
        else:
            print("No local model found. Cannot start server.")
            input("Press Enter to exit...")
            exit(1)
else:
    # User wants Local version
    if os.path.exists("docs/model.onnx") and os.path.exists("data/tokenizer.json"):
        onnx_path = "docs/model.onnx"
        tokenizer_path = "data/tokenizer.json"
    else:
        print("\nError: Local model (docs/model.onnx) or tokenizer (data/tokenizer.json) not found.")
        fallback = input("Would you like to download and use the Hugging Face version instead? (y/n): ").strip().lower()
        if fallback in ('y', 'yes'):
            model_ok = download_file(model_url, "docs/model_hf.onnx", "CooLM ONNX model")
            tokenizer_ok = download_file(tokenizer_url, "data/tokenizer_hf.json", "CooLM Tokenizer")
            if model_ok and tokenizer_ok:
                onnx_path = "docs/model_hf.onnx"
                tokenizer_path = "data/tokenizer_hf.json"
            else:
                print("Failed to download Hugging Face model files. Cannot start server.")
                input("Press Enter to exit...")
                exit(1)
        else:
            print("Please train your model first using train_coolm.bat before using the local version.")
            input("Press Enter to exit...")
            exit(1)

# Initialize engine with resolved paths
engine = None
try:
    engine = CooONNXInference(onnx_path=onnx_path, tokenizer_path=tokenizer_path)
except Exception as e:
    print(f"Error initializing ONNX inference: {e}")
    engine = None

class ChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve the UI static files from the docs/ folder
        super().__init__(*args, directory="docs", **kwargs)

    def do_POST(self):
        if self.path == '/chat':
            if not engine:
                self.send_error(500, "Inference engine failed to load.")
                return
            
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                user_text = data.get('text', '').strip()
                if not user_text:
                    response_text = "coo..."
                else:
                    # Run inference using the loaded ONNX model
                    response_text = engine.chat(user_text)
                
                # Send JSON response back to frontend
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'response': response_text}).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        else:
            self.send_error(404, "Not Found")

def run(port=8000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, ChatHandler)
    url = f"http://localhost:{port}/index.html"
    
    print(f"\n========================================")
    print(f" Server running at: {url}")
    print(f" Press Ctrl+C to stop the server.")
    print(f"========================================\n")
    
    # Open the browser automatically
    def open_browser():
        webbrowser.open(url)
        
    threading.Timer(0.5, open_browser).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run()
