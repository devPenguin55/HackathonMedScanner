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


@app.route("/api/submit", methods=["POST"])
def api_submit():
    """Receive everything the checker form collected.

    The front end POSTs a JSON payload of the questionnaire answers plus
    the uploaded X-ray's metadata (never the file bytes). Wire this up to
    the real backend / datastore / model here. For now we just acknowledge
    receipt so the client flow works end to end.
    """
    payload = request.get_json(silent=True) or {}
    user = session.get("user")
    answer_count = len(payload.get("answers") or {})
    has_file = bool(payload.get("file"))
    # TODO: forward `payload` to the backend (queue, DB, or model service).
    print(f"[submit] user={user!r} answers={answer_count} xray_attached={has_file}", flush=True)
    return jsonify({"ok": True, "received": True})


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main"))



import torch
import torch.nn as nn
from torchvision import models

# sqlite3.
m = getattr(models, "resnet18")(weights="DEFAULT")     # load pretrained weights
# The pretrained head predicts 1000 ImageNet classes. Replace it with a
# single output + Sigmoid so the model returns P(pneumonia) in [0, 1].
m.fc = nn.Sequential(
    nn.Linear(m.fc.in_features, 1),
    nn.Sigmoid()
)

device = "cuda" if torch.cuda.is_available() else "cpu"
m.to(device)

m.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
m.eval()
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),                       # fixed input size
    transforms.ToTensor(),                               # image -> tensor (grid of numbers) in [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet mean...
                         std=[0.229, 0.224, 0.225])    # ...and std
])
image_paths = [
    "test/0004cfab-14fd-4e49-80ba-63a80b6bddd6.jpg",
    "test/01be392f-a46d-4aef-a57e-9cd1a80dd47e.jpg",
]
images = []
for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {path}")

images = torch.stack(images).to(device)
with torch.no_grad():  
    outputs = m(images)  
    outputs = outputs.squeeze(1).cpu().numpy()
    print(outputs)


if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=1628, debug=False)
