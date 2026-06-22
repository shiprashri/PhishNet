import re
import urllib.parse
import base64
import tldextract

# Use cached snapshot only — no live network fetch needed
_extractor = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

# Known URL shorteners
SHORTENERS = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
              'is.gd', 'buff.ly', 'adf.ly', 'shorte.st', 'tiny.cc']

# Suspicious keywords commonly found in phishing URLs
PHISH_HINTS = ['login', 'signin', 'verify', 'secure', 'account',
               'update', 'banking', 'confirm', 'password', 'support',
               'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
               'urgent', 'suspend', 'validate', 'click', 'free']

# Known bad TLDs often used in phishing
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz',
                   '.top', '.club', '.work', '.click', '.link']

# Known brands for spoofing detection
BRANDS = ['paypal', 'google', 'facebook', 'apple', 'amazon',
          'microsoft', 'netflix', 'instagram', 'twitter', 'sbi',
          'hdfc', 'icici', 'ebay', 'dropbox', 'linkedin']


def extract_features(url):
    """
    Extract numerical features from a raw URL string.
    Returns a dict matching the training dataset columns.
    """
    features = {}

    # Parse URL components
    try:
        parsed = urllib.parse.urlparse(url)
        ext = _extractor(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        scheme = parsed.scheme.lower()
        full_url = url.lower()
        query = parsed.query.lower()
    except Exception:
        # Return all zeros if URL is malformed
        return {k: 0 for k in _get_feature_keys()}

    # ── URL LENGTH FEATURES ──────────────────────────────
    features['length_url'] = len(url)
    features['length_hostname'] = len(domain)

    # ── IP ADDRESS CHECK ─────────────────────────────────
    ip_pattern = re.compile(
        r'(\d{1,3}\.){3}\d{1,3}'
    )
    features['ip'] = 1 if ip_pattern.search(domain) else 0

    # ── SPECIAL CHARACTER COUNTS ─────────────────────────
    features['nb_dots']        = url.count('.')
    features['nb_hyphens']     = url.count('-')
    features['nb_at']          = url.count('@')
    features['nb_qm']          = url.count('?')
    features['nb_and']         = url.count('&')
    features['nb_eq']          = url.count('=')
    features['nb_underscore']  = url.count('_')
    features['nb_tilde']       = url.count('~')
    features['nb_percent']     = url.count('%')
    features['nb_slash']       = url.count('/')
    features['nb_star']        = url.count('*')
    features['nb_colon']       = url.count(':')
    features['nb_comma']       = url.count(',')
    features['nb_semicolumn']  = url.count(';')
    features['nb_dollar']      = url.count('$')
    features['nb_space']       = url.count(' ') + url.count('%20')

    # ── DOMAIN SPECIFIC ──────────────────────────────────
    features['nb_www']     = full_url.count('www')
    features['nb_com']     = full_url.count('.com')
    features['nb_dslash']  = url.count('//')

    features['http_in_path']  = 1 if 'http' in path else 0
    features['https_token']   = 1 if 'https' in domain else 0

    # Digit ratio
    digits_url  = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in domain)
    features['ratio_digits_url']  = digits_url / len(url) if url else 0
    features['ratio_digits_host'] = digits_host / len(domain) if domain else 0

    # Punycode (internationalized domain spoofing)
    features['punycode'] = 1 if 'xn--' in domain else 0

    # Port in URL
    features['port'] = 1 if parsed.port and parsed.port not in [80, 443] else 0

    # TLD in wrong place
    tld = '.' + ext.suffix if ext.suffix else ''
    features['tld_in_path']      = 1 if tld and tld in path else 0
    features['tld_in_subdomain'] = 1 if tld and tld in ext.subdomain else 0

    # Abnormal subdomain (long or many parts)
    subdomains = ext.subdomain.split('.') if ext.subdomain else []
    features['abnormal_subdomain'] = 1 if len(ext.subdomain) > 20 else 0
    features['nb_subdomains']      = len(subdomains)

    # Prefix-suffix (hyphen in domain)
    features['prefix_suffix'] = 1 if '-' in ext.domain else 0

    # Random-looking domain (high consonant cluster)
    features['random_domain'] = 1 if _is_random_domain(ext.domain) else 0

    # URL shortener
    features['shortening_service'] = 1 if any(s in full_url for s in SHORTENERS) else 0

    # File extension in path
    bad_exts = ['.exe', '.zip', '.rar', '.js', '.php', '.asp']
    features['path_extension'] = 1 if any(path.endswith(e) for e in bad_exts) else 0

    # ── REDIRECT FEATURES ────────────────────────────────
    features['nb_redirection']          = url.count('//') - 1 if '//' in url else 0
    features['nb_external_redirection'] = 1 if 'redirect' in full_url or 'redir' in full_url else 0

    # ── WORD / TEXT FEATURES ─────────────────────────────
    words = re.split(r'[\W_]+', full_url)
    words = [w for w in words if w]
    word_lengths = [len(w) for w in words]

    features['length_words_raw']   = len(words)
    features['char_repeat']        = _max_char_repeat(full_url)
    features['shortest_words_raw'] = min(word_lengths) if word_lengths else 0
    features['longest_words_raw']  = max(word_lengths) if word_lengths else 0
    features['avg_words_raw']      = sum(word_lengths) / len(word_lengths) if word_lengths else 0

    host_words   = re.split(r'[\W_]+', domain)
    host_words   = [w for w in host_words if w]
    host_lengths = [len(w) for w in host_words]
    features['shortest_word_host'] = min(host_lengths) if host_lengths else 0
    features['longest_word_host']  = max(host_lengths) if host_lengths else 0
    features['avg_word_host']      = sum(host_lengths) / len(host_lengths) if host_lengths else 0

    path_words   = re.split(r'[\W_]+', path)
    path_words   = [w for w in path_words if w]
    path_lengths = [len(w) for w in path_words]
    features['shortest_word_path'] = min(path_lengths) if path_lengths else 0
    features['longest_word_path']  = max(path_lengths) if path_lengths else 0
    features['avg_word_path']      = sum(path_lengths) / len(path_lengths) if path_lengths else 0

    # ── PHISHING HINT KEYWORDS ───────────────────────────
    features['phish_hints'] = sum(1 for h in PHISH_HINTS if h in full_url)

    # ── BRAND SPOOFING ───────────────────────────────────
    features['domain_in_brand']    = 1 if any(b in ext.domain for b in BRANDS) else 0
    features['brand_in_subdomain'] = 1 if any(b in ext.subdomain for b in BRANDS) else 0
    features['brand_in_path']      = 1 if any(b in path for b in BRANDS) else 0

    # Suspicious TLD
    features['suspecious_tld'] = 1 if tld in SUSPICIOUS_TLDS else 0

    # Statistical report placeholder (can't check live — default 0)
    features['statistical_report'] = 0

    # ── PAGE CONTENT FEATURES (not available from URL alone) ─
    # These were pre-computed in the dataset by visiting pages.
    # For a live URL we default to neutral/average values.
    features['nb_hyperlinks']         = 10
    features['ratio_intHyperlinks']   = 0.5
    features['ratio_extHyperlinks']   = 0.5
    features['ratio_nullHyperlinks']  = 0.1
    features['nb_extCSS']             = 2
    features['ratio_intRedirection']  = 0.1
    features['ratio_extRedirection']  = 0.1
    features['ratio_intErrors']       = 0.05
    features['ratio_extErrors']       = 0.05
    features['login_form']            = 0
    features['external_favicon']      = 0
    features['links_in_tags']         = 0.5
    features['submit_email']          = 0
    features['ratio_intMedia']        = 0.5
    features['ratio_extMedia']        = 0.5
    features['sfh']                   = 0
    features['iframe']                = 0
    features['popup_window']          = 0
    features['safe_anchor']           = 0.5
    features['onmouseover']           = 0
    features['right_clic']            = 0
    features['empty_title']           = 0
    features['domain_in_title']       = 1
    features['domain_with_copyright'] = 0

    # ── EXTERNAL REPUTATION (neutral averages from dataset) ─
    # We can't fetch live reputation; use dataset midpoints so model
    # relies on URL-structure features rather than unknown reputation.
    features['whois_registered_domain']    = 0
    features['domain_registration_length'] = 0
    features['domain_age']                 = 4000   # midpoint of legit/phish avg
    features['web_traffic']                = 0
    features['dns_record']                 = 0
    features['google_index']               = 0.5    # midpoint
    features['page_rank']                  = 3.0    # midpoint

    # ── BASE64 HIDDEN REDIRECT DETECTION ─────────────────
    hidden_redirect = _detect_base64_redirect(url)
    features['has_hidden_redirect'] = 1 if hidden_redirect else 0

    return features


def _detect_base64_redirect(url):
    """Check for base64-encoded redirect URLs hidden in query params."""
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        for key, values in params.items():
            for val in values:
                if len(val) < 8:
                    continue
                try:
                    padded = val + '=' * (-len(val) % 4)
                    decoded = base64.urlsafe_b64decode(padded).decode('utf-8', errors='ignore')
                    if decoded.startswith(('http://', 'https://')):
                        return decoded
                except Exception:
                    continue
    except Exception:
        pass
    return None


def analyze_upi_link(upi_string):
    """
    Lightweight risk check for UPI payment deep links (upi://pay?...).
    These aren't URLs, so the main extract_features() model can't score
    them — this is a separate, much simpler rule-based check instead.
    Returns a dict describing what was found.
    """
    findings = []
    risk = 0

    try:
        parsed = urllib.parse.urlparse(upi_string)
        params = urllib.parse.parse_qs(parsed.query)
    except Exception:
        return {
            'is_upi': True, 'risk_level': 'UNKNOWN', 'findings':
            ['Could not parse the UPI link parameters'],
            'payee_vpa': None, 'payee_name': None
        }

    vpa  = params.get('pa', [''])[0]
    name = params.get('pn', [''])[0]
    amount = params.get('am', [''])[0]

    # A valid VPA looks like name@bank — flag if the shape is missing or odd
    if not vpa or '@' not in vpa:
        findings.append('Payee VPA is missing or malformed')
        risk += 2
    else:
        handle = vpa.split('@')[-1].lower()
        # A handful of legitimate, well-known PSP handles
        known_handles = ['ybl', 'ibl', 'axl', 'okhdfcbank', 'okicici',
                          'okaxis', 'oksbi', 'paytm', 'apl', 'upi']
        if handle not in known_handles:
            findings.append(f'Payee uses an unfamiliar UPI handle (@{handle})')
            risk += 1

    if name:
        if any(b in name.lower() for b in BRANDS):
            findings.append(f'Payee name references a known brand ("{name}") \u2014 verify this is the real merchant')
            risk += 1
    else:
        findings.append('No payee name included with this payment request')
        risk += 1

    if amount:
        try:
            if float(amount) <= 0:
                findings.append('Requested amount is zero or invalid')
                risk += 1
        except ValueError:
            findings.append('Requested amount is not a valid number')
            risk += 1

    if not findings:
        findings.append('No notable suspicious patterns detected in this payment link')

    risk_level = 'HIGH' if risk >= 3 else 'MEDIUM' if risk >= 1 else 'LOW'

    return {
        'is_upi'    : True,
        'risk_level': risk_level,
        'findings'  : findings,
        'payee_vpa' : vpa or None,
        'payee_name': name or None,
        'amount'    : amount or None
    }


# ── HELPER FUNCTIONS ─────────────────────────────────────

def _is_random_domain(domain):
    """Check if domain looks randomly generated."""
    if not domain:
        return False
    vowels = set('aeiou')
    consonants = 0
    streak = 0
    for c in domain.lower():
        if c.isalpha() and c not in vowels:
            streak += 1
            consonants += 1
            if streak >= 4:
                return True
        else:
            streak = 0
    # Also flag if digit ratio is high
    digits = sum(c.isdigit() for c in domain)
    return digits / len(domain) > 0.4 if domain else False


def _max_char_repeat(text):
    """Find max consecutive character repeat."""
    if not text:
        return 0
    max_r = 1
    cur_r = 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            cur_r += 1
            max_r = max(max_r, cur_r)
        else:
            cur_r = 1
    return max_r


def _get_feature_keys():
    """Return list of all feature keys (for fallback)."""
    return extract_features('http://example.com').keys()