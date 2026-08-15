from flask import Flask, send_file
import os

app = Flask(__name__)
app.secret_key = os.urandom(32)  # clears all sessions


@app.route("/")
def main():
    return send_file("home.html")

if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=8000, debug=False)
