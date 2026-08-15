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
# CURATED_PROVIDERS = []

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


# The questionnaire, described to the model so it can weigh the answers.
QUESTIONS_TEXT = """QUESTIONS = [
    {id:'duration', q:'How long have you had the cough?'},
    {id:'abx', q:'Have you taken antibiotics for this, and did they help?'},
    {id:'constitutional', q:'Any night sweats, unusual fatigue, or unexplained weight loss?'},
    {id:'nodosum', q:'Any new painful red bumps on your shins (erythema nodosum)?'},
    {id:'exposure', q:'Do you live in, or have you traveled to, Arizona, Central California, or Nevada in the past 3 months?'},
    {id:'risk', q:'Any higher-risk factor (pregnancy, diabetes, immunocompromise, certain ancestries)?'},
    {id:'emergency', q:'Right now, severe headache with stiff neck, confusion, or trouble breathing?'},
]"""


def _pneumonia_score(file_meta):
    """Run the pneumonia model on the uploaded X-ray if we can find it locally.

    The client sends only the file name (not the bytes), so this works for the
    bundled sample images under test/. Returns a float in [0, 1] or None.
    """
    if not file_meta or not file_meta.get("name"):
        return None
    path = os.path.join("test", os.path.basename(file_meta["name"]))
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    batch = torch.stack([transform(img)]).to(device)
    with torch.no_grad():
        out = m(batch).squeeze(1).cpu().numpy()
    return round(float(out[0]), 4)


def _parse_assessment(text):
    """Extract {detection, confidence, reasoning} from the model's JSON reply."""
    t = (text or "").strip()
    # Strip a ```json ... ``` fence if the model wrapped its output in one.
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if "```" in t:
            t = t[:t.rfind("```")]
        t = t.strip()
    try:
        data = json.loads(t)
        if isinstance(data, dict):
            conf = data.get("confidence")
            try:
                conf = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf = None
            return {
                "detection": data.get("detectionOfValleyFever"),
                "confidence": conf,
                "reasoning": data.get("reasoning"),
            }
    except Exception:
        pass
    # Tolerant fallback if the reply is not clean JSON.
    out = {"detection": None, "confidence": None, "reasoning": None}
    try:
        out["detection"] = text.split('"detectionOfValleyFever"')[1].split(':', 1)[1].split(',')[0].strip().strip('"{} ')
    except Exception:
        pass
    try:
        out["confidence"] = float(text.split('"confidence"')[1].split(':', 1)[1].split(',')[0].strip().strip('"{} '))
    except Exception:
        pass
    try:
        out["reasoning"] = text.split('"reasoning"')[1].split(':', 1)[1].rsplit('"', 1)[0].strip().strip('"{} ')
    except Exception:
        pass
    return out


@app.route("/api/submit", methods=["POST"])
def api_submit():
    """Receive the checker answers plus the X-ray metadata, run the pneumonia
    model and Gemini reasoning, and return the assessment to the client.

    The whole assessment pipeline is guarded: if the model image is missing,
    or Gemini is unavailable, we still return HTTP 200 with assessment=null so
    the results screen (including the clinician page and doctor list) renders.
    """
    payload = request.get_json(silent=True) or {}
    user = session.get("user")
    answers = payload.get("answers") or {}
    file_meta = payload.get("file") or None

    assessment = None
    error = None
    try:
        pneumonia_score = _pneumonia_score(file_meta)

        wanted = '{ "detectionOfValleyFever": "yes" or "no", "confidence": 0.0 to 1.0, "reasoning": "text" }'
        prompt = f"""These are the questions asked: {QUESTIONS_TEXT}

I have a dict of the patient's answers to those questions. I also have a score
from our pneumonia detection model: closer to 1.0 (>= 0.5) means pneumonia,
below 0.5 means not pneumonia.

Weigh the answers together with the model's pneumonia score to reach the most
accurate estimate of whether the patient has valley fever. Symptoms alone do
not confirm valley fever, but they contribute to it.

Keep the reasoning field concise and reference the specific answers.

Answers: {answers}
Pneumonia detection score: {pneumonia_score}

Return only JSON in exactly this format:
{wanted}"""

        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        resp = client.models.generate_content(model=model_name, contents=prompt)
        parsed = _parse_assessment(resp.text)
        assessment = {
            "detection": parsed.get("detection"),
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
            "pneumoniaScore": pneumonia_score,
        }
    except Exception as e:
        error = str(e)
        print(f"[submit] assessment error: {e}", flush=True)

    print(f"[submit] user={user!r} answers={len(answers)} xray={bool(file_meta)} "
          f"assessment={'ok' if assessment else 'none'}", flush=True)
    return jsonify({"ok": True, "assessment": assessment, "error": error})


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main"))


if __name__ == '__main__':
    print("RUN")
    app.run(host="0.0.0.0", port=1628, debug=False)
