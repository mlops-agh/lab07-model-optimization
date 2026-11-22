import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def run(embed, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({"embedding": embed(body["text"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
