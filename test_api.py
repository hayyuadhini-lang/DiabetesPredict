
import requests

url = "http://127.0.0.1:8000/api/predict/"

data = {
    "age": 47,
    "gender": "Female",
    "city": "Mumbai",
    "bmi": 26.5,
    "family_history_diabetes": "Yes",
    "physical_activity_level": "Sedentary",
    "diet_type": "Vegetarian",
    "smoking_status": "Never",
    "alcohol_consumption": "Never",
    "hours_sleep_per_night": 9.0,
    "stress_level": 8,
    "fasting_blood_sugar": 178,
    "hba1c_level": 7.3,
    "blood_pressure_systolic": 146,
    "blood_pressure_diastolic": 90,
    "waist_circumference_cm": 107.1,
    "income_bracket": "High"
}

response = requests.post(url, json=data)

print("Status HTTP:", response.status_code)
print("Hasil API:")
print(response.json())
