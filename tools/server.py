import os
import json
import webbrowser
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from tools.onnx_chat import CooONNXInference

# Initialize the inference engine once, optimized for reuse.
print("Starting CooLM Local Web Server...")
try:
    engine = CooONNXInference(onnx_path="docs/model.onnx", tokenizer_path="data/tokenizer.json")
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
