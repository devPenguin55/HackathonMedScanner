from flask import Flask, send_file, request, redirect, url_for, session, jsonify
import os
import json
import urllib.request
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash

NPPES_URL = "https://npiregistry.cms.hhs.gov/api/"

# Shown when the live NPPES lookup is unreachable, so the results screen
# always has something real to point people to.
CURATED_PROVIDERS = [
    {"name": "Adriana T Gaidici MD", "specialty": "Infectious Disease",
     "address": "", "city": "Phoenix", "state": "AZ", "zip": "85008", "phone": "602-254-1136"},
    {"name": "Ahmad Salameh MD", "specialty": "Infectious Disease",
     "address": "", "city": "Phoenix", "state": "AZ", "zip": "85027", "phone": "602-439-0274"},
    {"name": "Valley Fever Center for Excellence (University of Arizona)",
     "specialty": "Coccidioidomycosis referral & clinician locator",
     "address": "1656 E Mabel St", "city": "Tucson", "state": "AZ", "zip": "85724", "phone": ""},
    {"name": "Sonora Quest Laboratories",
     "specialty": "Serology blood draw · patient service centers statewide",
     "address": "", "city": "Phoenix", "state": "AZ", "zip": "", "phone": ""},
]

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


def _nppes_lookup(prefix, taxonomy):
    """Query the federal NPPES registry for providers near a ZIP prefix."""
    query = urllib.parse.urlencode({
        "version": "2.1",
        "taxonomy_description": taxonomy,
        "postal_code": prefix,
        "limit": "20",
    })
    req = urllib.request.Request(NPPES_URL + "?" + query, headers={"User-Agent": "MedScan/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())

    out = []
    for r in data.get("results", []):
        b = r.get("basic", {})
        name = b.get("organization_name") or (b.get("first_name", "") + " " + b.get("last_name", "")).strip()
        if not name:
            continue
        addrs = r.get("addresses", []) or []
        loc = next((a for a in addrs if a.get("address_purpose") == "LOCATION"), addrs[0] if addrs else {})
        taxes = r.get("taxonomies", []) or []
        spec = next((t.get("desc") for t in taxes if t.get("primary")), taxes[0].get("desc") if taxes else taxonomy)
        out.append({
            "name": name.strip(),
            "specialty": spec or taxonomy,
            "address": loc.get("address_1", ""),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "zip": (loc.get("postal_code", "") or "")[:5],
            "phone": loc.get("telephone_number", ""),
        })
    return out


@app.route("/api/providers")
def api_providers():
    zip_ = (request.args.get("zip", "") or "").strip()
    if not (len(zip_) == 5 and zip_.isdigit()):
        return jsonify({"source": "error", "message": "Enter a 5-digit ZIP code.", "providers": []}), 400

    prefix = zip_[:3] + "*"
    providers = []
    try:
        for taxonomy in ("Infectious Disease", "Pulmonary Disease"):
            if len(providers) >= 12:
                break
            providers.extend(_nppes_lookup(prefix, taxonomy))
        # de-duplicate by name + zip
        seen, uniq = set(), []
        for p in providers:
            key = (p["name"], p["zip"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        providers = uniq[:12]
    except Exception:
        providers = []

    if providers:
        return jsonify({"source": "nppes", "providers": providers})
    return jsonify({"source": "fallback", "providers": CURATED_PROVIDERS})


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main"))


if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=1628, debug=False)
