"""
FarmGuard v2 — Smart Agriculture System
Real-time Login (SQLite) + Real-time Farmer Location (Geolocation API)
Dataset source: datasets/FarmGuard_Datasets.xlsx (5 sheets)
"""

# ── Flask core
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
from flask import session

# ── Flask extensions
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# ── Password hashing
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash

# ── ML — models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ── ML — preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ── ML — utilities
from sklearn.model_selection import train_test_split

# ── Data handling
from pandas import read_excel
from numpy import array
from numpy import argsort
from numpy import max as np_max

# ── Standard library
from os import makedirs
from os import path
from functools import wraps
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')

# ── Excel dataset path (single source of truth)
EXCEL_PATH = path.join('datasets', 'FarmGuard_Datasets.xlsx')

# ─── App Setup ───────────────────────────────────────────────────
import os
app = Flask(__name__)
app.config['SECRET_KEY'] = 'farmguard_ultra_secure_key_2026_xK9mP'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upload limit

CORS(app)
db = SQLAlchemy(app)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Database Models ──────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(120), nullable=False)
    email        = db.Column(db.String(200), unique=True, nullable=False)
    password     = db.Column(db.String(256), nullable=False)
    farm_name    = db.Column(db.String(200), default='My Farm')
    farm_area    = db.Column(db.String(100), default='Unknown')
    phone        = db.Column(db.String(20), default='')
    is_admin     = db.Column(db.Boolean, default=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    last_login   = db.Column(db.DateTime)
    # Location fields — updated via Geolocation API
    latitude     = db.Column(db.Float, default=None)
    longitude    = db.Column(db.Float, default=None)
    location_str = db.Column(db.String(300), default='Unknown')
    location_updated_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'farm_name': self.farm_name,
            'farm_area': self.farm_area,
            'phone': self.phone,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'location_str': self.location_str,
            'location_updated_at': self.location_updated_at.isoformat() if self.location_updated_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'is_admin': self.is_admin
        }


class AlertLog(db.Model):
    __tablename__ = 'alert_logs'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'))
    alert_id   = db.Column(db.Integer)
    pest       = db.Column(db.String(100))
    risk_level = db.Column(db.String(20))
    action     = db.Column(db.String(50), default='acknowledged')
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'))
    pred_type   = db.Column(db.String(50))   # crop / fertilizer / yield
    result      = db.Column(db.String(200))
    confidence  = db.Column(db.Float)
    latitude    = db.Column(db.Float)
    longitude   = db.Column(db.Float)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


# ─── Auth Decorator ───────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        # Verify user still exists in DB (handles stale sessions after DB reset)
        user = db.session.get(User, session['user_id'])
        if user is None:
            session.clear()
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def current_user():
    if 'user_id' not in session:
        return None
    user = db.session.get(User, session['user_id'])
    if user is None:
        session.clear()
    return user


# ─── Admin Auth Decorator ─────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('admin_login'))
        admin = db.session.get(User, session['admin_id'])
        if admin is None or not admin.is_admin:
            session.pop('admin_id', None)
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


def current_admin():
    if 'admin_id' not in session:
        return None
    admin = db.session.get(User, session['admin_id'])
    if admin is None or not admin.is_admin:
        session.pop('admin_id', None)
        return None
    return admin


# ─── Load Excel Sheets Once at Startup ───────────────────────────
print(f"📂 Loading datasets from: {EXCEL_PATH}")
_xl = pd.read_excel(EXCEL_PATH, sheet_name=None)   # loads all sheets into dict

# Map friendly names to actual sheet names in the workbook
def _sheet(name):
    """Return DataFrame for sheet whose name contains `name` (case-insensitive)."""
    for k, df in _xl.items():
        if name.lower() in k.lower():
            return df.copy()
    raise KeyError(f"Sheet containing '{name}' not found. Sheets: {list(_xl.keys())}")

print(f"   ✅ Sheets loaded: {list(_xl.keys())}")

# ─── Train ML Models ─────────────────────────────────────────────
def train_crop_model():
    # Reads from '🌾 Crop Data' sheet
    df = _sheet('Crop Data')
    print(f"   🌾 Crop Data    — {len(df)} rows, {df['label'].nunique()} crops")
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label_enc']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr)
    mdl = RandomForestClassifier(n_estimators=150, random_state=42)
    mdl.fit(X_s, y_tr)
    return mdl, scaler, le


def train_fertilizer_model():
    # Reads from '🧪 Fertilizer Data' sheet
    df = _sheet('Fertilizer Data')
    print(f"   🧪 Fertilizer   — {len(df)} rows, {df['Fertilizer_Name'].nunique()} types")
    le_s = LabelEncoder()
    le_c = LabelEncoder()
    le_f = LabelEncoder()
    df['soil_enc'] = le_s.fit_transform(df['Soil_Type'])
    df['crop_enc'] = le_c.fit_transform(df['Crop_Type'])
    df['fert_enc'] = le_f.fit_transform(df['Fertilizer_Name'])
    X = df[['Temparature', 'Humidity', 'Moisture', 'soil_enc', 'crop_enc',
            'Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['fert_enc']
    mdl = RandomForestClassifier(n_estimators=150, random_state=42)
    mdl.fit(X, y)
    return mdl, le_s, le_c, le_f


def train_yield_model():
    # Reads from '📈 Yield Data' sheet
    df = _sheet('Yield Data')
    print(f"   📈 Yield Data   — {len(df)} rows, {df['Crop'].nunique()} crops")
    le_se = LabelEncoder()
    le_cr = LabelEncoder()
    le_st = LabelEncoder()
    df['season_enc'] = le_se.fit_transform(df['Season'])
    df['crop_enc']   = le_cr.fit_transform(df['Crop'])
    df['state_enc']  = le_st.fit_transform(df['State'])
    X = df[['Area', 'season_enc', 'crop_enc', 'state_enc', 'Crop_Year']]
    y = df['Yield_kg_per_hectare']
    mdl = RandomForestRegressor(n_estimators=150, random_state=42)
    mdl.fit(X, y)
    return mdl, le_se, le_cr, le_st


print("🌱 Training ML models from Excel dataset...")
crop_model,  crop_scaler, crop_le              = train_crop_model()
fert_model,  fert_le_s, fert_le_c, fert_le_f  = train_fertilizer_model()
yield_model, yield_le_se, yield_le_cr, yield_le_st = train_yield_model()

# Load pest data for reference (read from '🚨 Pest Risk Data' sheet)
df_pest_ref = _sheet('Pest Risk')
print(f"   🚨 Pest Data    — {len(df_pest_ref)} rows, {df_pest_ref['pest_type'].nunique()} pests")
print("✅ All models trained from Excel — ready!")


# ─── Seed default users ───────────────────────────────────────────
def seed_users():
    # ── Seed admin account ─────────────────────────────────────────
    if not User.query.filter_by(email='admin@farmguard.com').first():
        admin = User(
            name='FarmGuard Admin',
            email='admin@farmguard.com',
            password=generate_password_hash('admin123'),
            farm_name='Admin Office',
            farm_area='N/A',
            phone='+91-00000-00000',
            is_admin=True
        )
        db.session.add(admin)

    # ── Seed demo farmer accounts ──────────────────────────────────
    demo_seeds = [
        dict(name='Rajesh Kumar',   email='farmer1@farmguard.com', password='password123',
             farm_name='Green Valley Farm',  farm_area='45 acres', phone='+91-98765-43210'),
        dict(name='Demo Farmer',    email='demo@farmguard.com',    password='demo123',
             farm_name='Demo Smart Farm',    farm_area='20 acres', phone='+91-99999-00000'),
        dict(name='Priya Sharma',   email='priya@farmguard.com',   password='priya123',
             farm_name='Sunrise Agricultural Fields', farm_area='30 acres', phone='+91-91234-56789'),
    ]
    for s in demo_seeds:
        if not User.query.filter_by(email=s['email']).first():
            u = User(name=s['name'], email=s['email'],
                     password=generate_password_hash(s['password']),
                     farm_name=s['farm_name'], farm_area=s['farm_area'], phone=s['phone'])
            db.session.add(u)
    db.session.commit()


# ─── Pages ────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        email    = data.get('email', '').strip().lower()
        password = data.get('password', '')

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'success': False, 'message': 'Invalid email or password.'})

        user.last_login = datetime.utcnow()
        db.session.commit()

        session.permanent = True
        session['user_id'] = user.id
        return jsonify({'success': True, 'name': user.name, 'redirect': '/dashboard'})

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        name      = data.get('name', '').strip()
        email     = data.get('email', '').strip().lower()
        password  = data.get('password', '')
        farm_name = data.get('farm_name', 'My Farm')
        farm_area = data.get('farm_area', '')
        phone     = data.get('phone', '')

        if not name or not email or not password:
            return jsonify({'success': False, 'message': 'Name, email and password are required.'})
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered. Please log in.'})

        user = User(name=name, email=email,
                    password=generate_password_hash(password),
                    farm_name=farm_name, farm_area=farm_area, phone=phone)
        db.session.add(user)
        db.session.commit()

        session['user_id'] = user.id
        return jsonify({'success': True, 'name': user.name, 'redirect': '/dashboard'})

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user())


@app.route('/crop-prediction')
@login_required
def crop_prediction():
    return render_template('crop_prediction.html', user=current_user())


@app.route('/fertilizer')
@login_required
def fertilizer():
    return render_template('fertilizer.html', user=current_user())


@app.route('/yield-forecast')
@login_required
def yield_forecast():
    return render_template('yield_forecast.html', user=current_user())


@app.route('/pest-alerts')
@login_required
def pest_alerts():
    return render_template('pest_alerts.html', user=current_user())


@app.route('/datasets')
@login_required
def datasets():
    """Dataset viewer — shows all Excel sheets in the browser."""
    return render_template('datasets.html', user=current_user())


@app.route('/api/dataset/<sheet_name>')
@login_required
def api_dataset(sheet_name):
    """Return rows from a named Excel sheet as JSON for the dataset viewer."""
    sheet_map = {
        'crop':       'Crop Data',
        'fertilizer': 'Fertilizer Data',
        'yield':      'Yield Data',
        'pest':       'Pest Risk',
        'summary':    'Summary',
    }
    key = sheet_map.get(sheet_name.lower())
    if not key:
        return jsonify({'success': False, 'error': 'Unknown sheet'})
    try:
        df = _sheet(key)
        # Replace NaN with empty string for JSON safety
        df = df.fillna('')
        return jsonify({
            'success':  True,
            'sheet':    sheet_name,
            'columns':  list(df.columns),
            'rows':     df.values.tolist(),
            'total':    len(df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/download-excel')
@login_required
def download_excel():
    """Let logged-in users download the Excel dataset file."""
    from flask import send_file
    return send_file(
        EXCEL_PATH,
        as_attachment=True,
        download_name='FarmGuard_Datasets.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user())


# ─── Auth API ────────────────────────────────────────────────────
@app.route('/api/me', methods=['GET'])
@login_required
def api_me():
    return jsonify({'success': True, 'user': current_user().to_dict()})


# ─── Location API (Manual) ────────────────────────────────────────
@app.route('/api/update-location', methods=['POST'])
@login_required
def update_location():
    """
    Accepts manual location from the farmer.
    Supports two modes:
      1. Text-based  : state + district + village  (geocoded via Nominatim)
      2. Coordinates : latitude + longitude + location_str
    """
    data    = request.get_json(silent=True) or {}
    lat     = data.get('latitude')
    lng     = data.get('longitude')
    loc_str = data.get('location_str', '').strip()

    # ── Mode 1: coords already provided (map click / geocode result)
    if lat is not None and lng is not None:
        user = current_user()
        user.latitude            = float(lat)
        user.longitude           = float(lng)
        user.location_str        = loc_str or f"{float(lat):.4f}°N, {float(lng):.4f}°E"
        user.location_updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True,
                        'location': user.location_str,
                        'latitude': user.latitude,
                        'longitude': user.longitude})

    # ── Mode 2: text address → geocode via Nominatim
    village  = data.get('village', '').strip()
    district = data.get('district', '').strip()
    state    = data.get('state', '').strip()

    if not state:
        return jsonify({'success': False, 'message': 'Please select at least a State.'})

    # Build human-readable string
    parts    = [p for p in [village, district, state, 'India'] if p]
    loc_str  = ', '.join(parts)

    # Try Nominatim geocoding (free, no API key)
    geo_lat, geo_lng = None, None
    try:
        import urllib.request
        import urllib.parse
        import json as _json
        query   = urllib.parse.quote(loc_str)
        url     = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1&countrycodes=in"
        req     = urllib.request.Request(url, headers={'User-Agent': 'FarmGuard/2.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = _json.loads(resp.read())
        if results:
            geo_lat = float(results[0]['lat'])
            geo_lng = float(results[0]['lon'])
    except Exception:
        pass   # Geocoding failed — still save text location

    # If geocoding failed, use state capital coordinates as fallback
    STATE_COORDS = {
        'Andaman and Nicobar':     (11.7401, 92.6586),
        'Andhra Pradesh':          (15.9129, 79.7400),
        'Arunachal Pradesh':       (28.2180, 94.7278),
        'Assam':                   (26.2006, 92.9376),
        'Bihar':                   (25.0961, 85.3131),
        'Chandigarh':              (30.7333, 76.7794),
        'Chhattisgarh':            (21.2787, 81.8661),
        'Dadra and Nagar Haveli':  (20.1809, 73.0169),
        'Daman and Diu':           (20.3974, 72.8328),
        'Delhi':                   (28.6139, 77.2090),
        'Goa':                     (15.2993, 74.1240),
        'Gujarat':                 (22.2587, 71.1924),
        'Haryana':                 (29.0588, 76.0856),
        'Himachal Pradesh':        (31.1048, 77.1734),
        'Jammu and Kashmir':       (33.7782, 76.5762),
        'Jharkhand':               (23.6102, 85.2799),
        'Karnataka':               (15.3173, 75.7139),
        'Kerala':                  (10.8505, 76.2711),
        'Ladakh':                  (34.2268, 77.5619),
        'Lakshadweep':             (10.5667, 72.6417),
        'Madhya Pradesh':          (22.9734, 78.6569),
        'Maharashtra':             (19.7515, 75.7139),
        'Manipur':                 (24.6637, 93.9063),
        'Meghalaya':               (25.4670, 91.3662),
        'Mizoram':                 (23.1645, 92.9376),
        'Nagaland':                (26.1584, 94.5624),
        'Odisha':                  (20.9517, 85.0985),
        'Puducherry':              (11.9416, 79.8083),
        'Punjab':                  (31.1048, 75.8122),
        'Rajasthan':               (27.0238, 74.2179),
        'Sikkim':                  (27.5330, 88.5122),
        'Tamil Nadu':              (11.1271, 78.6569),
        'Telangana':               (18.1124, 79.0193),
        'Tripura':                 (23.9408, 91.9882),
        'Uttar Pradesh':           (26.8467, 80.9462),
        'Uttarakhand':             (30.0668, 79.0193),
        'West Bengal':             (22.9868, 87.8550),
    }
    if geo_lat is None:
        geo_lat, geo_lng = STATE_COORDS.get(state, (20.5937, 78.9629))

    user = current_user()
    user.latitude            = geo_lat
    user.longitude           = geo_lng
    user.location_str        = loc_str
    user.location_updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({'success': True,
                    'location': loc_str,
                    'latitude': geo_lat,
                    'longitude': geo_lng,
                    'geocoded': True})


@app.route('/api/farmer-location', methods=['GET'])
@login_required
def farmer_location():
    user = current_user()
    return jsonify({
        'success': True,
        'latitude':  user.latitude,
        'longitude': user.longitude,
        'location_str': user.location_str,
        'updated_at': user.location_updated_at.isoformat() if user.location_updated_at else None
    })


# ─── Crop Prediction API ──────────────────────────────────────────
@app.route('/api/crop-predict', methods=['POST'])
@login_required
def api_crop_predict():
    try:
        data = request.get_json()
        features = np.array([[float(data['N']), float(data['P']), float(data['K']),
                               float(data['temperature']), float(data['humidity']),
                               float(data['ph']), float(data['rainfall'])]])
        fs = crop_scaler.transform(features)
        pred  = crop_model.predict(fs)[0]
        proba = crop_model.predict_proba(fs)[0]
        crop  = crop_le.inverse_transform([pred])[0]

        top3 = [{'crop': crop_le.inverse_transform([i])[0].capitalize(),
                 'confidence': round(float(proba[i])*100, 1)}
                for i in np.argsort(proba)[-3:][::-1]]

        info_map = {
            'rice':     {'season':'Kharif',      'duration':'120-150 days','water':'High'},
            'maize':    {'season':'Kharif/Rabi',  'duration':'80-110 days', 'water':'Medium'},
            'wheat':    {'season':'Rabi',         'duration':'100-150 days','water':'Medium'},
            'cotton':   {'season':'Kharif',       'duration':'150-180 days','water':'Medium'},
            'sugarcane':{'season':'Year-round',   'duration':'300-365 days','water':'Very High'},
        }
        info = info_map.get(crop.lower(), {'season':'Kharif/Rabi','duration':'90-120 days','water':'Medium'})

        # Log prediction
        user = current_user()
        db.session.add(PredictionLog(user_id=user.id, pred_type='crop',
                                     result=crop, confidence=round(float(max(proba))*100,1),
                                     latitude=user.latitude, longitude=user.longitude))
        db.session.commit()

        return jsonify({'success':True, 'primary_crop':crop.capitalize(),
                        'confidence':round(float(max(proba))*100,1),
                        'recommendations':top3, 'crop_info':info})
    except Exception as e:
        return jsonify({'success':False, 'error':str(e)})


# ─── Fertilizer API ───────────────────────────────────────────────
@app.route('/api/fertilizer-predict', methods=['POST'])
@login_required
def api_fertilizer_predict():
    try:
        data = request.get_json()
        soils  = ['Sandy','Loamy','Black','Red','Clayey']
        crops  = ['Maize','Sugarcane','Cotton','Tobacco','Paddy','Barley',
                  'Wheat','Millets','Oil seeds','Pulses','Ground Nuts']
        s_enc = soils.index(data['soil_type'])  if data['soil_type']  in soils  else 0
        c_enc = crops.index(data['crop_type'])  if data['crop_type']  in crops  else 0

        feat = np.array([[float(data['temperature']), float(data['humidity']),
                          float(data['moisture']), s_enc, c_enc,
                          float(data['nitrogen']), float(data['potassium']),
                          float(data['phosphorous'])]])
        pred  = fert_model.predict(feat)[0]
        fert  = fert_le_f.inverse_transform([pred])[0]

        fert_info = {
            'Urea':     {'npk':'46-0-0',  'description':'High nitrogen for leafy growth','application':'50-100 kg/ha','timing':'Before sowing or top dressing','color':'#4CAF50'},
            'DAP':      {'npk':'18-46-0', 'description':'Excellent for root development','application':'75-125 kg/ha','timing':'At time of sowing','color':'#2196F3'},
            '14-35-14': {'npk':'14-35-14','description':'Balanced NPK for flowering crops','application':'100-150 kg/ha','timing':'During vegetative stage','color':'#FF9800'},
            '28-28':    {'npk':'28-28-0', 'description':'Nitrogen and phosphorus complex','application':'80-120 kg/ha','timing':'Early growth stage','color':'#9C27B0'},
            '17-17-17': {'npk':'17-17-17','description':'Complete balanced fertilizer','application':'100-200 kg/ha','timing':'Throughout season','color':'#F44336'},
        }
        info = fert_info.get(fert, {'npk':'Balanced','description':'Recommended for optimal growth',
                                     'application':'100 kg/ha','timing':'As per soil test','color':'#607D8B'})

        n = float(data['nitrogen']); k = float(data['potassium']); p = float(data['phosphorous'])
        nd = {
            'nitrogen':    'Low' if n<20 else 'Adequate' if n<40 else 'High',
            'potassium':   'Low' if k<10 else 'Adequate' if k<25 else 'High',
            'phosphorous': 'Low' if p<15 else 'Adequate' if p<35 else 'High',
        }

        user = current_user()
        db.session.add(PredictionLog(user_id=user.id, pred_type='fertilizer',
                                     result=fert, latitude=user.latitude, longitude=user.longitude))
        db.session.commit()

        return jsonify({'success':True,'fertilizer':fert,'info':info,'nutrient_deficiency':nd})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})


# ─── Yield Forecast API ───────────────────────────────────────────
@app.route('/api/yield-forecast', methods=['POST'])
@login_required
def api_yield_forecast():
    try:
        data    = request.get_json()
        area    = float(data['area'])
        season  = data['season']
        crop    = data['crop']
        state   = data['state']
        year    = int(data.get('year', 2025))

        seasons = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn']
        crops   = ['Rice','Wheat','Maize','Cotton','Sugarcane','Pulses',
                   'Groundnut','Soybean','Sunflower','Barley','Jowar','Bajra']
        states  = sorted([
            'Andaman and Nicobar','Andhra Pradesh','Arunachal Pradesh','Assam',
            'Bihar','Chandigarh','Chhattisgarh','Dadra and Nagar Haveli',
            'Daman and Diu','Delhi','Goa','Gujarat','Haryana','Himachal Pradesh',
            'Jammu and Kashmir','Jharkhand','Karnataka','Kerala','Ladakh',
            'Lakshadweep','Madhya Pradesh','Maharashtra','Manipur','Meghalaya',
            'Mizoram','Nagaland','Odisha','Puducherry','Punjab','Rajasthan',
            'Sikkim','Tamil Nadu','Telangana','Tripura','Uttar Pradesh',
            'Uttarakhand','West Bengal'
        ])

        s_enc  = seasons.index(season) if season in seasons else 0
        c_enc  = crops.index(crop)     if crop   in crops   else 0
        st_enc = states.index(state)   if state  in states  else 0

        feat = np.array([[area, s_enc, c_enc, st_enc, year]])
        yph  = float(yield_model.predict(feat)[0])
        total = yph * area

        monthly = [{'month':m,'yield':round(yph*random.uniform(0.85,1.15),0)}
                   for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]

        comp = {'last_year':  round(yph*random.uniform(0.90,0.98),2),
                'national_avg':round(yph*random.uniform(0.85,0.95),2),
                'projected':   round(yph,2)}
        factors = {'soil_health': random.randint(70,95),
                   'weather_suitability': random.randint(65,90),
                   'pest_risk': random.randint(10,40),
                   'market_demand': random.randint(60,95)}

        user = current_user()
        db.session.add(PredictionLog(user_id=user.id, pred_type='yield',
                                     result=f'{crop} {round(yph,0)} kg/ha',
                                     latitude=user.latitude, longitude=user.longitude))
        db.session.commit()

        return jsonify({'success':True,'yield_per_hectare':round(yph,2),
                        'total_production':round(total,2),'unit':'kg',
                        'monthly_forecast':monthly,'comparison':comp,'factors':factors})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})


# ─── Pest Alerts API ──────────────────────────────────────────────
@app.route('/api/pest-alerts')
@login_required
def api_pest_alerts():
    # Pull real pest records from the loaded Excel sheet
    sample = df_pest_ref.sample(min(8, len(df_pest_ref)), random_state=random.randint(0, 9999))
    colors = {'High': '#FF4444', 'Medium': '#FF9800', 'Low': '#4CAF50'}
    recs   = {
        'High':   'Immediate action required! Apply targeted pesticide within 24-48 hours.',
        'Medium': 'Monitor closely. Apply preventive treatment in 3-5 days.',
        'Low':    'Low risk. Continue regular field monitoring.'
    }
    alerts = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        rl   = str(row['risk_level'])
        pest = str(row['pest_type'])
        crop = str(row['affected_crop'])
        alerts.append({
            'id':           i,
            'pest':         pest,
            'crop':         crop,
            'risk_level':   rl,
            'color':        colors.get(rl, '#888'),
            'temperature':  round(float(row['temperature']), 1),
            'humidity':     round(float(row['humidity']),    1),
            'rainfall':     round(float(row['rainfall']),    1),
            'wind_speed':   round(float(row['wind_speed']),  1),
            'timestamp':    (datetime.now() - timedelta(hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M'),
            'recommendation': recs.get(rl, 'Monitor field conditions.') + f' (Pest: {pest})',
            'affected_area': f'{random.randint(5, 50)} acres'
        })

    alerts.sort(key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}.get(x['risk_level'], 3))
    weather = {
        'temperature': round(random.uniform(25, 35), 1),
        'humidity':    round(random.uniform(65, 88), 1),
        'rainfall':    round(random.uniform(0,  25), 1),
        'wind_speed':  round(random.uniform(8,  28), 1),
        'condition':   random.choice(['Partly Cloudy', 'Sunny', 'Overcast', 'Light Rain'])
    }
    return jsonify({
        'success':      True,
        'alerts':       alerts,
        'weather':      weather,
        'total_high':   sum(1 for a in alerts if a['risk_level'] == 'High'),
        'total_medium': sum(1 for a in alerts if a['risk_level'] == 'Medium'),
        'total_low':    sum(1 for a in alerts if a['risk_level'] == 'Low')
    })


@app.route('/api/acknowledge-alert', methods=['POST'])
@login_required
def acknowledge_alert():
    data = request.get_json(silent=True) or {}
    user = current_user()
    log  = AlertLog(user_id=user.id,
                    alert_id=data.get('alert_id'),
                    pest=data.get('pest',''),
                    risk_level=data.get('risk_level',''),
                    action='acknowledged')
    db.session.add(log)
    db.session.commit()
    return jsonify({'success':True,'message':'Alert acknowledged and saved to database.'})


# ─── Pest Image Analysis ──────────────────────────────────────────
@app.route('/api/analyze-pest-image', methods=['POST'])
@login_required
def analyze_pest_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ── Extract real image features using Pillow ──────────────────
    try:
        from PIL import Image as PILImage
        img = PILImage.open(filepath).convert('RGB')
        img_small = img.resize((64, 64))          # downsample for speed
        pixels    = list(img_small.getdata())      # list of (R,G,B) tuples

        total_px  = len(pixels)
        avg_r     = sum(p[0] for p in pixels) / total_px
        avg_g     = sum(p[1] for p in pixels) / total_px
        avg_b     = sum(p[2] for p in pixels) / total_px
        brightness = (avg_r + avg_g + avg_b) / 3   # 0-255

        # Green-channel dominance → leaf/plant health indicator
        green_ratio  = avg_g / (avg_r + avg_g + avg_b + 1)
        # Yellow indicator (high R+G, low B)
        yellow_ratio = (avg_r + avg_g) / (2 * (avg_b + 1))
        # Brown indicator (R dominant, low G+B)
        brown_ratio  = avg_r / (avg_g + avg_b + 1)
        # Dark patches → high contrast variance
        variance = sum((p[0]-avg_r)**2 + (p[1]-avg_g)**2 + (p[2]-avg_b)**2
                       for p in pixels) / total_px

        # Use image file size as additional entropy seed
        file_size = os.path.getsize(filepath)
        # Deterministic seed from actual image content so same image → same result
        img_seed  = int(avg_r * 1000 + avg_g * 100 + avg_b * 10 + (file_size % 97))

    except Exception:
        # Fallback: derive seed from filename bytes
        img_seed     = sum(ord(c) for c in file.filename)
        brightness   = 128
        green_ratio  = 0.35
        yellow_ratio = 1.0
        brown_ratio  = 1.0
        variance     = 1000

    # ── Full pest catalogue with image-feature scoring ────────────
    ALL_PESTS = {
        'Aphids': {
            'trigger': lambda: green_ratio * 80 + (brightness / 255) * 20,
            'affected': ['Wheat', 'Mustard', 'Tomato', 'Potato'],
            'severity_thresholds': (75, 55),
            'color': '#4CAF50',
            'recommendations': [
                'Apply neem oil spray (2–3 ml/L water) every 7 days',
                'Install yellow sticky traps at crop canopy level',
                'Introduce ladybird beetles (Coccinella) as biological control',
                'Apply imidacloprid 0.5 ml/L if infestation >30% leaves',
                'Remove heavily infested shoots and destroy them',
            ],
        },
        'Whitefly': {
            'trigger': lambda: yellow_ratio * 50 + (255 - brightness) / 255 * 50,
            'affected': ['Cotton', 'Tomato', 'Chilli', 'Brinjal'],
            'severity_thresholds': (70, 50),
            'color': '#FF9800',
            'recommendations': [
                'Apply insecticidal soap (5 ml/L) on leaf undersides',
                'Use silver/reflective mulches to repel adults',
                'Spray acetamiprid 0.3 g/L water at 10-day intervals',
                'Install yellow sticky traps (10/acre)',
                'Remove and destroy heavily infested leaves',
            ],
        },
        'Brown Plant Hopper': {
            'trigger': lambda: brown_ratio * 60 + (variance / 5000) * 40,
            'affected': ['Rice', 'Paddy'],
            'severity_thresholds': (72, 52),
            'color': '#795548',
            'recommendations': [
                'Drain field for 3–5 days to expose hoppers to sun',
                'Apply buprofezin 25 SC @ 1 ml/L water',
                'Avoid excess nitrogen fertilizer application',
                'Use light traps to monitor adult population',
                'Maintain 2.5 cm water level during vegetative stage',
            ],
        },
        'Bollworm': {
            'trigger': lambda: brown_ratio * 40 + (1 - green_ratio) * 60,
            'affected': ['Cotton', 'Tomato', 'Chickpea'],
            'severity_thresholds': (68, 48),
            'color': '#F44336',
            'recommendations': [
                'Apply chlorpyrifos 2 ml/L at egg-hatching stage',
                'Install pheromone traps (5/acre) for monitoring',
                'Spray Bt (Bacillus thuringiensis) 2 g/L as biocontrol',
                'Pick and destroy affected bolls/fruits manually',
                'Use HaNPV (nuclear polyhedrosis virus) 250 LE/ha',
            ],
        },
        'Leaf Folder': {
            'trigger': lambda: green_ratio * 50 + (variance / 4000) * 50,
            'affected': ['Rice', 'Paddy'],
            'severity_thresholds': (65, 45),
            'color': '#8BC34A',
            'recommendations': [
                'Apply chlorantraniliprole 18.5 SC @ 0.3 ml/L',
                'Use light traps at night to catch adult moths',
                'Clip folded leaf tips to destroy larvae inside',
                'Spray cartap hydrochloride 50 SP @ 1 g/L',
                'Maintain field hygiene by removing crop residues',
            ],
        },
        'Stem Borer': {
            'trigger': lambda: (brightness / 255) * 40 + brown_ratio * 30 + (variance / 6000) * 30,
            'affected': ['Rice', 'Maize', 'Sugarcane'],
            'severity_thresholds': (70, 50),
            'color': '#FF5722',
            'recommendations': [
                'Apply carbofuran 3G granules @ 10 kg/acre at tillering',
                'Collect and destroy egg masses on leaves',
                'Release Trichogramma parasitoids (50,000/ha)',
                'Spray monocrotophos 1.6 ml/L at whorl stage',
                'Cut and burn infested tillers showing dead hearts',
            ],
        },
        'Thrips': {
            'trigger': lambda: (1 - green_ratio) * 50 + yellow_ratio * 30 + (img_seed % 20),
            'affected': ['Cotton', 'Onion', 'Chilli', 'Groundnut'],
            'severity_thresholds': (65, 45),
            'color': '#9C27B0',
            'recommendations': [
                'Apply spinosad 45 SC @ 0.3 ml/L water',
                'Install blue sticky traps at 10/acre',
                'Spray dimethoate 2 ml/L in early morning',
                'Maintain proper spacing to improve air circulation',
                'Remove and destroy crop debris after harvest',
            ],
        },
        'Armyworm': {
            'trigger': lambda: (variance / 3000) * 50 + brown_ratio * 30 + (img_seed % 20),
            'affected': ['Maize', 'Sorghum', 'Wheat', 'Rice'],
            'severity_thresholds': (73, 53),
            'color': '#607D8B',
            'recommendations': [
                'Apply lambda-cyhalothrin 5 EC @ 0.6 ml/L at dusk',
                'Set up pheromone traps (5/acre) for early detection',
                'Use Bt spray (2 g/L) on young larvae (<3rd instar)',
                'Deep plough after harvest to kill pupae in soil',
                'Hand-pick and destroy egg masses and larvae',
            ],
        },
        'Mealybug': {
            'trigger': lambda: (1 - green_ratio) * 40 + (brightness / 255) * 40 + (img_seed % 20),
            'affected': ['Sugarcane', 'Papaya', 'Grapes', 'Mango'],
            'severity_thresholds': (67, 47),
            'color': '#E91E63',
            'recommendations': [
                'Apply profenofos 50 EC @ 2 ml/L as stem treatment',
                'Release Cryptolaemus montrouzieri beetles (10/plant)',
                'Scrub infested stems with brushes and neem soap',
                'Apply imidacloprid soil drench at base of plant',
                'Remove ant colonies which protect mealybugs',
            ],
        },
        'Cutworm': {
            'trigger': lambda: (1 - brightness / 255) * 55 + brown_ratio * 25 + (img_seed % 20),
            'affected': ['Maize', 'Potato', 'Tomato', 'Cabbage'],
            'severity_thresholds': (66, 46),
            'color': '#FF6F00',
            'recommendations': [
                'Apply chlorpyrifos 20 EC @ 2.5 ml/L as soil drench',
                'Use bran bait mixed with insecticide near plant base',
                'Install collar barriers (5 cm) around seedlings',
                'Deep cultivation destroys pupae and exposes larvae',
                'Apply Steinernema nematodes as biological control',
            ],
        },
    }

    # ── Score every pest against this specific image ──────────────
    rng_local = random.Random(img_seed)    # deterministic per image

    scored = []
    for pest_name, info in ALL_PESTS.items():
        base_score  = info['trigger']()
        # Add small controlled noise (deterministic per pest+image combo)
        noise_seed  = img_seed + sum(ord(c) for c in pest_name)
        noise_rng   = random.Random(noise_seed)
        noise       = noise_rng.uniform(-8, 8)
        raw_score   = max(10, min(98, base_score + noise))
        scored.append((pest_name, round(raw_score, 1), info))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # ── Top 3 detections ──────────────────────────────────────────
    top3 = scored[:3]
    # Normalise so top pest gets 65-95 range, others proportionally lower
    top_raw   = top3[0][1]
    top_conf  = round(rng_local.uniform(70, 94), 1)
    scale     = top_conf / top_raw if top_raw > 0 else 1

    detected = []
    for i, (pname, raw, info) in enumerate(top3):
        conf = round(raw * scale, 1) if i > 0 else top_conf
        conf = max(15, min(97, conf))

        hi_thresh, lo_thresh = info['severity_thresholds']
        severity = ('High' if conf >= hi_thresh else
                    'Moderate' if conf >= lo_thresh else 'Low')
        detected.append({
            'pest':        pname,
            'confidence':  conf,
            'severity':    severity,
            'color':       info['color'],
            'affected_crops': info['affected'],
        })

    primary = detected[0]

    # ── Crop at risk from primary pest ────────────────────────────
    primary_info     = ALL_PESTS[primary['pest']]
    affected_crops   = primary_info['affected']
    recommendations  = primary_info['recommendations']

    return jsonify({
        'success':        True,
        'detected_pests': detected,
        'primary_pest':   primary['pest'],
        'confidence':     primary['confidence'],
        'severity':       primary['severity'],
        'affected_crops': affected_crops,
        'recommendations':recommendations,
        'image_features': {
            'brightness':   round(brightness, 1),
            'green_ratio':  round(green_ratio * 100, 1),
            'yellow_ratio': round(yellow_ratio, 2),
            'variance':     round(variance, 1),
        },
    })


# ─── Dashboard Stats ──────────────────────────────────────────────
@app.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    user = current_user()
    recent_preds = PredictionLog.query.filter_by(user_id=user.id)\
                                      .order_by(PredictionLog.timestamp.desc()).limit(5).all()
    recent_alerts= AlertLog.query.filter_by(user_id=user.id)\
                                 .order_by(AlertLog.timestamp.desc()).limit(3).all()

    activity = []
    for p in recent_preds:
        icons = {'crop':'🌾','fertilizer':'🧪','yield':'📈'}
        activity.append({'action':f'{p.pred_type.capitalize()} prediction: {p.result}',
                         'time': _time_ago(p.timestamp),
                         'icon': icons.get(p.pred_type,'📊')})
    for a in recent_alerts:
        activity.append({'action':f'Alert acknowledged: {a.pest or "pest alert"}',
                         'time': _time_ago(a.timestamp), 'icon':'⚠️'})
    if not activity:
        activity = [{'action':'Welcome to FarmGuard!','time':'Just now','icon':'🌿'},
                    {'action':'Run your first crop prediction','time':'','icon':'🌾'},
                    {'action':'Check pest risk alerts','time':'','icon':'🚨'}]

    return jsonify({
        'active_alerts':   random.randint(2,8),
        'crop_health':     random.randint(72,95),
        'soil_moisture':   round(random.uniform(35,65),1),
        'predicted_yield': f'{random.randint(3200,4800)} kg/ha',
        'location': {'lat':user.latitude,'lng':user.longitude,'str':user.location_str},
        'weather': {'temp':round(random.uniform(22,38),1),
                    'humidity':round(random.uniform(55,85),1),
                    'condition':random.choice(['Sunny','Partly Cloudy','Overcast','Light Rain'])},
        'recent_activity': activity[:6]
    })


def _time_ago(dt):
    diff = datetime.utcnow() - dt
    if diff.seconds < 60:      return 'Just now'
    if diff.seconds < 3600:    return f'{diff.seconds//60} min ago'
    if diff.days == 0:         return f'{diff.seconds//3600} hr ago'
    return f'{diff.days} day{"s" if diff.days>1 else ""} ago'


# ─── Profile Update ───────────────────────────────────────────────
@app.route('/api/update-profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json(silent=True) or {}
    user = current_user()
    if 'name'      in data: user.name      = data['name']
    if 'farm_name' in data: user.farm_name = data['farm_name']
    if 'farm_area' in data: user.farm_area = data['farm_area']
    if 'phone'     in data: user.phone     = data['phone']
    if 'new_password' in data and data['new_password']:
        if not check_password_hash(user.password, data.get('current_password','')):
            return jsonify({'success':False,'message':'Current password is incorrect.'})
        user.password = generate_password_hash(data['new_password'])
    db.session.commit()
    return jsonify({'success':True,'message':'Profile updated successfully.','user':user.to_dict()})


# ─── Edit Credentials Page ───────────────────────────────────────
@app.route('/edit-credentials')
@login_required
def edit_credentials():
    return render_template('edit_credentials.html', user=current_user())


@app.route('/api/edit-credentials', methods=['POST'])
@login_required
def api_edit_credentials():
    """
    Lets the logged-in user update:
      - name, email, phone
      - password (requires current password)
    """
    data = request.get_json(silent=True) or {}
    user = current_user()
    errors = []

    # ── Name
    name = data.get('name', '').strip()
    if name:
        user.name = name

    # ── Email (check uniqueness)
    new_email = data.get('email', '').strip().lower()
    if new_email and new_email != user.email:
        if User.query.filter(User.email == new_email, User.id != user.id).first():
            errors.append('That email address is already registered by another account.')
        else:
            user.email = new_email

    # ── Phone
    phone = data.get('phone', '').strip()
    if phone:
        user.phone = phone

    # ── Farm name & area
    if data.get('farm_name'): user.farm_name = data['farm_name'].strip()
    if data.get('farm_area'): user.farm_area = data['farm_area'].strip()

    # ── Password change
    new_pwd     = data.get('new_password', '').strip()
    confirm_pwd = data.get('confirm_password', '').strip()
    if new_pwd:
        if not check_password_hash(user.password, data.get('current_password', '')):
            errors.append('Current password is incorrect.')
        elif len(new_pwd) < 6:
            errors.append('New password must be at least 6 characters.')
        elif new_pwd != confirm_pwd:
            errors.append('New password and confirmation do not match.')
        else:
            user.password = generate_password_hash(new_pwd)

    if errors:
        return jsonify({'success': False, 'message': ' | '.join(errors)})

    db.session.commit()
    return jsonify({'success': True,
                    'message': 'Credentials updated successfully!',
                    'user': user.to_dict()})


# ══════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════

# ─── Admin Login ──────────────────────────────────────────────────
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        email    = data.get('email', '').strip().lower()
        password = data.get('password', '')
        user = User.query.filter_by(email=email, is_admin=True).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'success': False, 'message': 'Invalid admin credentials.'})
        user.last_login = datetime.utcnow()
        db.session.commit()
        session['admin_id'] = user.id
        return jsonify({'success': True, 'redirect': '/admin/dashboard'})
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    return redirect(url_for('admin_login'))


# ─── Admin Dashboard ──────────────────────────────────────────────
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', admin=current_admin())


@app.route('/api/admin/stats')
@admin_required
def api_admin_stats():
    total_farmers   = User.query.filter_by(is_admin=False).count()
    total_preds     = PredictionLog.query.count()
    total_alerts    = AlertLog.query.count()
    active_today    = User.query.filter(
        User.is_admin == False,
        User.last_login >= datetime.utcnow().replace(hour=0, minute=0, second=0)
    ).count()

    # Recent predictions (last 10)
    recent_preds = db.session.query(PredictionLog, User).join(
        User, PredictionLog.user_id == User.id
    ).order_by(PredictionLog.timestamp.desc()).limit(10).all()

    recent = [{
        'farmer': p.User.name,
        'type':   p.PredictionLog.pred_type,
        'result': p.PredictionLog.result,
        'confidence': round(p.PredictionLog.confidence or 0, 1),
        'time':   p.PredictionLog.timestamp.strftime('%d %b %Y %H:%M')
    } for p in recent_preds]

    # Prediction breakdown by type
    pred_types = db.session.query(
        PredictionLog.pred_type, db.func.count(PredictionLog.id)
    ).group_by(PredictionLog.pred_type).all()

    return jsonify({
        'success': True,
        'stats': {
            'total_farmers': total_farmers,
            'total_predictions': total_preds,
            'total_alerts': total_alerts,
            'active_today': active_today,
        },
        'recent_predictions': recent,
        'pred_breakdown': {t: c for t, c in pred_types}
    })


# ─── Admin Farmer Management ──────────────────────────────────────
@app.route('/admin/farmers')
@admin_required
def admin_farmers():
    return render_template('admin_farmers.html', admin=current_admin())


@app.route('/api/admin/farmers')
@admin_required
def api_admin_farmers():
    q = request.args.get('q', '').strip()
    query = User.query.filter_by(is_admin=False)
    if q:
        query = query.filter(
            (User.name.ilike(f'%{q}%')) |
            (User.email.ilike(f'%{q}%')) |
            (User.farm_name.ilike(f'%{q}%'))
        )
    farmers = query.order_by(User.created_at.desc()).all()
    result = []
    for f in farmers:
        pred_count  = PredictionLog.query.filter_by(user_id=f.id).count()
        alert_count = AlertLog.query.filter_by(user_id=f.id).count()
        result.append({
            **f.to_dict(),
            'prediction_count': pred_count,
            'alert_count': alert_count
        })
    return jsonify({'success': True, 'farmers': result, 'total': len(result)})


@app.route('/api/admin/farmers/<int:farmer_id>')
@admin_required
def api_admin_farmer_detail(farmer_id):
    farmer = db.session.get(User, farmer_id)
    if not farmer or farmer.is_admin:
        return jsonify({'success': False, 'message': 'Farmer not found.'})

    predictions = PredictionLog.query.filter_by(user_id=farmer_id)\
                    .order_by(PredictionLog.timestamp.desc()).limit(20).all()
    alerts = AlertLog.query.filter_by(user_id=farmer_id)\
                .order_by(AlertLog.timestamp.desc()).limit(10).all()

    return jsonify({
        'success': True,
        'farmer': farmer.to_dict(),
        'predictions': [{
            'type': p.pred_type, 'result': p.result,
            'confidence': round(p.confidence or 0, 1),
            'time': p.timestamp.strftime('%d %b %Y %H:%M')
        } for p in predictions],
        'alerts': [{
            'pest': a.pest, 'risk_level': a.risk_level,
            'action': a.action,
            'time': a.timestamp.strftime('%d %b %Y %H:%M')
        } for a in alerts]
    })


# ─── Admin Create Farmer ──────────────────────────────────────────
@app.route('/admin/farmers/create')
@admin_required
def admin_create_farmer():
    return render_template('admin_create_farmer.html', admin=current_admin())


@app.route('/api/admin/farmers/create', methods=['POST'])
@admin_required
def api_admin_create_farmer():
    data = request.get_json(silent=True) or {}
    name      = data.get('name', '').strip()
    email     = data.get('email', '').strip().lower()
    password  = data.get('password', '').strip()
    farm_name = data.get('farm_name', 'My Farm').strip()
    farm_area = data.get('farm_area', '').strip()
    phone     = data.get('phone', '').strip()

    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Name, email and password are required.'})
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'})

    farmer = User(
        name=name, email=email,
        password=generate_password_hash(password),
        farm_name=farm_name, farm_area=farm_area,
        phone=phone, is_admin=False
    )
    db.session.add(farmer)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Farmer {name} created successfully!', 'farmer': farmer.to_dict()})


# ─── Admin Edit Farmer ────────────────────────────────────────────
@app.route('/admin/farmers/<int:farmer_id>/edit')
@admin_required
def admin_edit_farmer(farmer_id):
    farmer = db.session.get(User, farmer_id)
    if not farmer or farmer.is_admin:
        return redirect(url_for('admin_farmers'))
    return render_template('admin_edit_farmer.html', admin=current_admin(), farmer=farmer)


@app.route('/api/admin/farmers/<int:farmer_id>/edit', methods=['POST'])
@admin_required
def api_admin_edit_farmer(farmer_id):
    farmer = db.session.get(User, farmer_id)
    if not farmer or farmer.is_admin:
        return jsonify({'success': False, 'message': 'Farmer not found.'})

    data = request.get_json(silent=True) or {}
    if data.get('name'):      farmer.name      = data['name'].strip()
    if data.get('phone'):     farmer.phone     = data['phone'].strip()
    if data.get('farm_name'): farmer.farm_name = data['farm_name'].strip()
    if data.get('farm_area'): farmer.farm_area = data['farm_area'].strip()

    new_email = data.get('email', '').strip().lower()
    if new_email and new_email != farmer.email:
        if User.query.filter(User.email == new_email, User.id != farmer_id).first():
            return jsonify({'success': False, 'message': 'Email already taken by another account.'})
        farmer.email = new_email

    new_pwd = data.get('new_password', '').strip()
    if new_pwd:
        if len(new_pwd) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})
        farmer.password = generate_password_hash(new_pwd)

    db.session.commit()
    return jsonify({'success': True, 'message': 'Farmer updated successfully!', 'farmer': farmer.to_dict()})


# ─── Admin Delete Farmer ──────────────────────────────────────────
@app.route('/api/admin/farmers/<int:farmer_id>/delete', methods=['POST'])
@admin_required
def api_admin_delete_farmer(farmer_id):
    farmer = db.session.get(User, farmer_id)
    if not farmer or farmer.is_admin:
        return jsonify({'success': False, 'message': 'Farmer not found.'})
    # Delete associated logs
    PredictionLog.query.filter_by(user_id=farmer_id).delete()
    AlertLog.query.filter_by(user_id=farmer_id).delete()
    name = farmer.name
    db.session.delete(farmer)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Farmer {name} deleted successfully.'})


# ─── Admin Change Own Password ────────────────────────────────────
@app.route('/api/admin/change-password', methods=['POST'])
@admin_required
def api_admin_change_password():
    data   = request.get_json(silent=True) or {}
    admin  = current_admin()
    cur    = data.get('current_password', '')
    new_p  = data.get('new_password', '').strip()
    conf   = data.get('confirm_password', '').strip()
    if not check_password_hash(admin.password, cur):
        return jsonify({'success': False, 'message': 'Current password is incorrect.'})
    if len(new_p) < 6:
        return jsonify({'success': False, 'message': 'New password must be at least 6 characters.'})
    if new_p != conf:
        return jsonify({'success': False, 'message': 'Passwords do not match.'})
    admin.password = generate_password_hash(new_p)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Admin password updated successfully.'})


# ─── Bootstrap DB & Run ───────────────────────────────────────────
with app.app_context():
    db.create_all()
    seed_users()
    print("✅ Database ready — tables created and seeded.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
