from flask import Flask, send_file, request, redirect, url_for
import os

app = Flask(__name__)
app.secret_key = os.urandom(32)  # clears all sessions


@app.route("/")
def main():
    return send_file("home.html")


@app.route("/login")
def login():
    return send_file("login.html")


@app.route("/loginData", methods=["POST"])
def login_data():
    # The login form (login.html) posts here.
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    remember = request.form.get("remember") == "1"

    # TODO: replace this stub with a real credential check (hashed lookup).
    # For now, no accounts exist, so every attempt is treated as invalid
    # and sent back to the form with an error flag.
    print(f"[loginData] attempt username={username!r} remember={remember}")

    authenticated = False
    if authenticated:
        return redirect(url_for("main"))
    return redirect(url_for("login", error=1))


if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=4782, debug=False)
