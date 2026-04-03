# Farm-Guard---Complete-Agriculture-Analytics-System
FarmGuard is a smart agriculture system that uses machine learning to help farmers make better decisions. It recommends suitable crops, fertilizers, and pest control measures based on soil and weather data. The system improves productivity, reduces resource wastage, and supports sustainable farming practices.
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser
http://localhost:5000
| Name         | Email                                                 | Password    |
| ------------ | ----------------------------------------------------- | ----------- |
| Demo Farmer  | [demo@farmguard.com](mailto:demo@farmguard.com)       | demo123     |
| Rajesh Kumar | [farmer1@farmguard.com](mailto:farmer1@farmguard.com) | password123 |
| Priya Sharma | [priya@farmguard.com](mailto:priya@farmguard.com)     | priya123    |


Real Authentication (SQLite Database)

Passwords hashed using Werkzeug PBKDF2-SHA256

User data stored in instance/farmguard.db

Session management using Flask

Registration creates real database accounts

Last login tracking

Prediction history stored per user

📍 Real-Time GPS Location
Browser Geolocation API for coordinates

OpenStreetMap Nominatim for reverse geocoding

Location stored in database for each user

Leaflet.js interactive maps

Live coordinates shown in UI

Status indicators:
Orange = Detecting
Green = Active
Red = Error

Security Features

Passwords hashed (PBKDF2-SHA256)

Protected routes using @login_required

Session-based authentication

SQL injection prevention via ORM

File upload size limited to 16MB
