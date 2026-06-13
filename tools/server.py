import os
import json
import webbrowser
import threading
import shutil
from http.server import SimpleHTTPRequestHandler, HTTPServer
from tools.onnx_chat import CooONNXInference

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

if choice == "1":
    # User wants Hugging Face version
    if os.path.exists("docs/model_hf.onnx") and os.path.exists("data/tokenizer_hf.json"):
        print("Using cached Hugging Face version.")
        onnx_path = "docs/model_hf.onnx"
        tokenizer_path = "data/tokenizer_hf.json"
    else:
        print("Hugging Face model files not found locally.")
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading model.onnx from Hugging Face...")
            cached_model = hf_hub_download(repo_id="MdHussain121/coolm-42M", filename="model.onnx")
            print("Downloading tokenizer.json from Hugging Face...")
            cached_tokenizer = hf_hub_download(repo_id="MdHussain121/coolm-42M", filename="tokenizer.json")
            
            os.makedirs("docs", exist_ok=True)
            os.makedirs("data", exist_ok=True)
            shutil.copy(cached_model, "docs/model_hf.onnx")
            shutil.copy(cached_tokenizer, "data/tokenizer_hf.json")
            print("Successfully downloaded Hugging Face model!")
            onnx_path = "docs/model_hf.onnx"
            tokenizer_path = "data/tokenizer_hf.json"
        except Exception as e:
            print(f"\nError downloading Hugging Face model: {e}")
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
            try:
                from huggingface_hub import hf_hub_download
                print("Downloading model.onnx from Hugging Face...")
                cached_model = hf_hub_download(repo_id="MdHussain121/coolm-42M", filename="model.onnx")
                print("Downloading tokenizer.json from Hugging Face...")
                cached_tokenizer = hf_hub_download(repo_id="MdHussain121/coolm-42M", filename="tokenizer.json")
                
                os.makedirs("docs", exist_ok=True)
                os.makedirs("data", exist_ok=True)
                shutil.copy(cached_model, "docs/model_hf.onnx")
                shutil.copy(cached_tokenizer, "data/tokenizer_hf.json")
                print("Successfully downloaded Hugging Face model!")
                onnx_path = "docs/model_hf.onnx"
                tokenizer_path = "data/tokenizer_hf.json"
            except Exception as e:
                print(f"Error downloading Hugging Face model: {e}")
                print("Cannot start server.")
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
