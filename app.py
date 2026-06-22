import os
import sys
import io
import base64
import time
import requests
import joblib
import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from utils.feature_extractor import extract_features, _detect_base64_redirect, analyze_upi_link

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────
MODEL_PATH   = os.path.join('model', 'phishing_model.pkl')
FEATURE_PATH = os.path.join('model', 'feature_cols.pkl')

# Get a free key at https://www.virustotal.com/gui/join-us
VT_API_KEY = os.environ.get('VT_API_KEY', '')

# Trusted domains — never flagged as phishing regardless of model output
WHITELIST = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.in', 'amazon.com',
    'github.com', 'microsoft.com', 'apple.com', 'linkedin.com',
    'wikipedia.org', 'stackoverflow.com', 'instagram.com', 'twitter.com',
    'x.com', 'whatsapp.com', 'netflix.com', 'flipkart.com', 'paypal.com',
    'sbi.co.in', 'hdfcbank.com', 'icicibank.com', 'gmail.com', 'yahoo.com',
    'reddit.com', 'chatgpt.com', 'openai.com', 'anthropic.com', 'claude.ai'
]

# Search engines — their result pages always carry long query strings
# (?q=, &form=, &sourceid=...) which the model misreads as suspicious.
# Real-world phishing rarely impersonates the search engine's own domain
# itself (it impersonates the *brand in the search result*), so the
# domain — not the query string — is what should decide legitimacy here.
SEARCH_ENGINES = [
    'google.com', 'bing.com', 'yahoo.com', 'duckduckgo.com',
    'yandex.com', 'baidu.com', 'ecosia.org', 'brave.com'
]

# In-memory scan history — most recent first. Resets on server restart.
# (No database needed for a demo-scale academic project.)
SCAN_HISTORY = []
MAX_HISTORY  = 25

# ── LOAD MODEL ────────────────────────────────────────────
model        = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_PATH)
print(f"Model loaded — {len(feature_cols)} features expected")


# ── HELPERS ───────────────────────────────────────────────
def is_whitelisted(url):
    import tldextract
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}".lower()
    return any(domain == w or domain.endswith('.' + w) for w in WHITELIST)


def is_search_engine(url):
    """True if the URL's domain is a known search engine."""
    import tldextract
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}".lower()
    return any(domain == s for s in SEARCH_ENGINES)


def check_virustotal(url):
    """Cross-check a URL against VirusTotal. Returns dict, never raises."""
    if not VT_API_KEY:
        return {'checked': False, 'reason': 'no_api_key'}
    try:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip('=')
        headers = {'x-apikey': VT_API_KEY}
        res = requests.get(
            f'https://www.virustotal.com/api/v3/urls/{url_id}',
            headers=headers, timeout=5
        )
        if res.status_code == 200:
            stats = res.json()['data']['attributes']['last_analysis_stats']
            malicious  = stats.get('malicious', 0)
            suspicious = stats.get('suspicious', 0)
            harmless   = stats.get('harmless', 0)
            return {
                'checked'   : True,
                'malicious' : malicious,
                'suspicious': suspicious,
                'harmless'  : harmless,
                'verdict'   : 'PHISHING' if malicious > 2 else 'LEGITIMATE'
            }
        return {'checked': False, 'reason': f'status_{res.status_code}'}
    except Exception as e:
        return {'checked': False, 'reason': str(e)}


def get_signals(features, hidden_redirect):
    """Human-readable list of suspicious URL signals."""
    signals = []
    if features.get('ip'):
        signals.append('IP address used instead of a domain name')
    if features.get('phish_hints', 0) > 0:
        signals.append(f"Suspicious keyword(s) found ({features['phish_hints']})")
    if features.get('prefix_suffix'):
        signals.append('Hyphen in domain name — common spoofing pattern')
    if features.get('nb_subdomains', 0) > 2:
        signals.append(f"Unusually high subdomain count ({features['nb_subdomains']})")
    if features.get('shortening_service'):
        signals.append('URL shortener detected — destination is hidden')
    if features.get('suspecious_tld'):
        signals.append('Free / commonly abused top-level domain')
    if features.get('punycode'):
        signals.append('Punycode encoding — possible domain spoofing')
    if features.get('https_token'):
        signals.append('"https" embedded in domain to fake legitimacy')
    if features.get('brand_in_subdomain'):
        signals.append('Known brand name placed in subdomain')
    if features.get('random_domain'):
        signals.append('Domain name looks randomly generated')
    if features.get('length_url', 0) > 100:
        signals.append(f"Unusually long URL ({features['length_url']} characters)")
    if features.get('nb_at', 0) > 0:
        signals.append('"@" symbol present — can redirect browsers silently')
    if hidden_redirect:
        signals.append(f"Hidden base64-encoded redirect found → {hidden_redirect}")
    if not signals:
        signals.append('No notable suspicious patterns detected')
    return signals


def decode_qr(image_bytes):
    """
    Decode a QR code from raw image bytes using a multi-strategy pipeline.

    Strategy order (fastest / most accurate first):
      1. WeChatQRCode — deep-learning based, handles blurry / rotated / small QR codes well
      2. QRCodeDetectorAruco — ArUco-based, good on perspective-distorted codes
      3. Standard QRCodeDetector on the original image
      4. Preprocessing pass: upscale + CLAHE + adaptive threshold, then retry 2 & 3
      5. Multi-scale scan: try smaller crops if the QR is only part of a screenshot

    Returns the decoded string, or None if no QR code was found.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    def _try_wechat(image):
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode()
            results, _ = detector.detectAndDecode(image)
            if results:
                return results[0]
        except Exception:
            pass
        return None

    def _try_aruco(image):
        try:
            detector = cv2.QRCodeDetectorAruco()
            data, _, _ = detector.detectAndDecode(image)
            if data:
                return data
        except Exception:
            pass
        return None

    def _try_standard(image):
        try:
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(image)
            if data:
                return data
        except Exception:
            pass
        return None

    def _preprocess(image):
        """Upscale small images, enhance contrast, return grayscale + sharpened versions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Upscale if image is very small
        if max(h, w) < 400:
            scale = 400 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold version
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to BGR for detectors that need it
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        thresh_bgr   = cv2.cvtColor(thresh,   cv2.COLOR_GRAY2BGR)
        return [enhanced_bgr, thresh_bgr]

    # ── Pass 1: try on original image ─────────────────────────────────────
    for fn in (_try_wechat, _try_aruco, _try_standard):
        result = fn(img)
        if result:
            return result

    # ── Pass 2: preprocess and retry ──────────────────────────────────────
    for variant in _preprocess(img):
        for fn in (_try_wechat, _try_aruco, _try_standard):
            result = fn(variant)
            if result:
                return result

    # ── Pass 3: multi-scale — QR might be a small part of a screenshot ────
    h, w = img.shape[:2]
    for scale in (0.75, 0.5, 1.25, 1.5):
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for fn in (_try_wechat, _try_aruco, _try_standard):
            result = fn(resized)
            if result:
                return result

    return None


def resolve_redirect(url, max_hops=5):
    """
    Follow HTTP redirects to find the real destination behind a
    shortened URL (bit.ly, tinyurl, etc.). Returns the final URL it
    landed on, or the original URL unchanged if resolution fails.
    Uses HEAD requests only — never downloads page content.
    """
    current = url
    try:
        for _ in range(max_hops):
            res = requests.head(current, allow_redirects=False, timeout=4,
                                 headers={'User-Agent': 'Mozilla/5.0 (PhishNet Scanner)'})
            if res.status_code in (301, 302, 303, 307, 308):
                next_url = res.headers.get('Location')
                if not next_url:
                    break
                if next_url.startswith('/'):
                    # relative redirect — resolve against current host
                    from urllib.parse import urljoin
                    next_url = urljoin(current, next_url)
                current = next_url
            else:
                break
    except Exception:
        pass
    return current


def analyze_url(url, origin='manual'):
    """
    Core detection pipeline shared by the text-input scan and the QR scan.
    Returns the full result dict and also appends a record to SCAN_HISTORY.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    original_url = url
    pre_features  = extract_features(url)
    was_shortened = bool(pre_features.get('shortening_service'))

    # If the URL uses a shortener, resolve where it actually leads and
    # score that real destination instead of the shortener wrapper.
    # The shortener fact itself is still surfaced as a finding below.
    if was_shortened:
        resolved = resolve_redirect(url)
        if resolved and resolved != url:
            url = resolved

    whitelisted   = is_whitelisted(url)
    search_engine = is_search_engine(url)
    features = extract_features(url)
    hidden_redirect = _detect_base64_redirect(url)

    row = {col: features.get(col, 0) for col in feature_cols}
    X = pd.DataFrame([row])

    model_pred  = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]
    phish_prob  = round(float(proba[1]) * 100, 2)
    legit_prob  = round(float(proba[0]) * 100, 2)

    vt = check_virustotal(url)

    # ── Decision logic: whitelist/search engine > VirusTotal > ML model ──
    source = 'model'
    if whitelisted or search_engine:
        final_pred, final_label = 0, 'LEGITIMATE'
        phish_prob, legit_prob = 1.0, 99.0
        source = 'search_engine' if search_engine and not whitelisted else 'whitelist'
    elif vt.get('checked'):
        if vt['malicious'] > 2:
            final_pred, final_label = 1, 'PHISHING'
            source = 'virustotal'
        elif vt['harmless'] > 10 and model_pred == 1:
            final_pred, final_label = 0, 'LEGITIMATE'
            source = 'virustotal'
        else:
            final_pred  = model_pred
            final_label = 'PHISHING' if model_pred == 1 else 'LEGITIMATE'
    else:
        final_pred  = model_pred
        final_label = 'PHISHING' if model_pred == 1 else 'LEGITIMATE'

    if hidden_redirect and final_pred == 0:
        final_label = 'SUSPICIOUS'

    signals = get_signals(features, hidden_redirect)
    if was_shortened:
        if url != original_url:
            signals.insert(0, f"Shortened link resolved to: {url}")
        else:
            # Resolution failed — remove the generic shortener line that
            # get_signals already added, replace with a clearer one.
            signals = [s for s in signals if 'URL shortener detected' not in s]
            signals.insert(0, "URL shortener detected — destination could not be resolved (scored on the shortened link itself)")

    result = {
        'url'          : url,
        'original_url' : original_url if original_url != url else None,
        'was_shortened': was_shortened,
        'prediction'   : final_pred,
        'label'        : final_label,
        'source'       : source,
        'confidence'   : round(max(phish_prob, legit_prob), 2),
        'phish_prob'   : phish_prob,
        'legit_prob'   : legit_prob,
        'signals'      : signals,
        'virustotal'   : vt,
        'whitelisted'  : whitelisted,
        'search_engine': search_engine,
        'origin'       : origin
    }

    SCAN_HISTORY.insert(0, {
        'url'      : url,
        'label'    : final_label,
        'confidence': result['confidence'],
        'origin'   : origin,
        'timestamp': time.strftime('%H:%M:%S')
    })
    del SCAN_HISTORY[MAX_HISTORY:]

    return result


# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        result = analyze_url(url, origin='manual')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-qr', methods=['POST'])
def predict_qr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        decoded = decode_qr(image_bytes)

        if not decoded:
            return jsonify({'error': 'No QR code detected in this image'}), 422

        # UPI payment links aren't web URLs, so they get their own
        # lightweight VPA/merchant check instead of the URL model.
        if decoded.lower().startswith('upi://'):
            upi_result = analyze_upi_link(decoded)
            return jsonify({
                'url'         : decoded,
                'decoded_text': decoded,
                'is_upi'      : True,
                'label'       : f"UPI PAYMENT LINK \u2014 {upi_result['risk_level']} RISK",
                'risk_level'  : upi_result['risk_level'],
                'payee_vpa'   : upi_result['payee_vpa'],
                'payee_name'  : upi_result['payee_name'],
                'amount'      : upi_result['amount'],
                'signals'     : upi_result['findings'],
                'origin'      : 'qr'
            })

        if not decoded.startswith(('http://', 'https://', 'www.')):
            # QR contains other non-URL data (text, WiFi config, contact card, etc.)
            return jsonify({
                'error'       : 'QR code does not contain a URL',
                'decoded_text': decoded
            }), 422

        result = analyze_url(decoded, origin='qr')
        result['decoded_text'] = decoded
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': SCAN_HISTORY})


if __name__ == '__main__':
    app.run(debug=True, port=5000)