# test_flow.py
import requests

API = "http://localhost:8000"

print("1. Simulating disaster...")
r = requests.post(f"{API}/simulate", json={"location": "eThekwini", "scenario": "flood"})
print(r.json())

print("2. Predicting risk...")
r = requests.post(f"{API}/predict")
print(f"Scored {len(r.json()['predictions'])} locations")

print("3. Generating alerts...")
r = requests.post(f"{API}/alerts/generate")
print(f"Generated {r.json()['generated']} alerts")