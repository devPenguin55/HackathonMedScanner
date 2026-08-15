from flask import Flask, send_file, request, redirect, url_for, session, jsonify
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(32)  # clears all sessions on restart

USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


@app.route("/")
def main():
    return send_file("home.html")


@app.route("/login")
def login():
    return send_file("login.html")


@app.route("/api/me")
def api_me():
    # The gated checker calls this to decide whether to unlock.
    return jsonify({"logged_in": "user" in session, "user": session.get("user")})


@app.route("/loginData", methods=["POST"])
def login_data():
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")

    users = load_users()
    stored = users.get(username)
    if stored and check_password_hash(stored, password):
        session["user"] = username
        return redirect(url_for("main") + "#check")

    # error=1 -> bad credentials
    return redirect(url_for("login", error=1))


@app.route("/registerData", methods=["POST"])
def register_data():
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")

    if not username or not password:
        return redirect(url_for("login", error=1, mode="register"))

    users = load_users()
    if username in users:
        # error=2 -> account already exists
        return redirect(url_for("login", error=2, mode="register"))

    # pbkdf2:sha256 avoids scrypt, which isn't available in every hashlib build
    users[username] = generate_password_hash(password, method="pbkdf2:sha256")
    save_users(users)
    session["user"] = username
    return redirect(url_for("main") + "#check")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main"))


if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=1628, debug=False)
