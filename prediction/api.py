
import os
import requests
import joblib
import pandas as pd

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import RiwayatPrediksi
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import renderer_classes
from django.utils import timezone


# Memuat model Machine Learning
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "model_diabetes.joblib"
)

model = joblib.load(MODEL_PATH)


# Fitur yang harus diterima API
FEATURES = [
    "age",
    "gender",
    "city",
    "bmi",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm",
    "income_bracket"
]


@api_view(["POST"])
def predict_diabetes(request):

    try:
        data = request.data

        # Memeriksa kelengkapan input
        missing_fields = [
            field for field in FEATURES
            if field not in data
        ]

        if missing_fields:
            return Response({
                "error": "Ada input yang belum dikirim.",
                "missing_fields": missing_fields
            }, status=status.HTTP_400_BAD_REQUEST)

        # Mengubah input menjadi DataFrame
        input_data = pd.DataFrame(
            [[data[field] for field in FEATURES]],
            columns=FEATURES
        )

        # Melakukan prediksi
        prediction = model.predict(input_data)[0]


        # Mengambil probabilitas prediksi
        probabilities = model.predict_proba(input_data)[0]


        # Menghubungkan nama kelas dengan probabilitas
        probability_result = {
            str(label): round(float(probability) * 100, 2)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        # Menyimpan hasil prediksi ke database
        RiwayatPrediksi.objects.create(
            age=int(data["age"]),
            gender=str(data["gender"]),
            bmi=float(data["bmi"]),
            prediction=str(prediction),
            low_probability=float(probability_result.get("Low", 0)),
            moderate_probability=float(probability_result.get("Moderate", 0)),
            high_probability=float(probability_result.get("High", 0))
        )

        return Response({
            "success": True,
            "prediction": str(prediction),
            "probabilities": probability_result,
            "message": "Prediksi berhasil dilakukan."
        })

    except (ValueError, TypeError) as error:
        return Response({
            "success": False,
            "error": f"Input tidak valid: {str(error)}"
        }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as error:
        return Response({
            "success": False,
            "error": f"Terjadi kesalahan: {str(error)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def explain_graph(request):

    try:
        # Mengambil data hasil prediksi dari frontend
        prediction = request.data.get("prediction")
        probabilities = request.data.get("probabilities")

        # Validasi input
        if not prediction or not isinstance(probabilities, dict):
            return Response({
                "success": False,
                "error": "Data prediksi atau probabilitas belum tersedia."
            }, status=status.HTTP_400_BAD_REQUEST)

        labels = ["Low", "Moderate", "High"]

        # Memastikan semua kategori memiliki nilai probabilitas
        if not all(label in probabilities for label in labels):
            return Response({
                "success": False,
                "error": "Data probabilitas harus memiliki Low, Moderate, dan High."
            }, status=status.HTTP_400_BAD_REQUEST)

        # Mengambil API Key dan nama model dari file .env
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )

        if not api_key:
            return Response({
                "success": False,
                "error": "GROQ_API_KEY belum dikonfigurasi."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Menyusun data probabilitas untuk dikirim ke Groq
        probability_text = "\n".join([
            f"{label}: {probabilities[label]}%"
            for label in labels
        ])

        # Prompt untuk Groq
        prompt = f"""
        Berikut adalah hasil prediksi dari model Machine Learning
        menggunakan algoritma Random Forest.

        Kategori prediksi tertinggi: {prediction}

        Hasil probabilitas:
        {probability_text}

        Tugas kamu:
        1. Jelaskan kategori dengan probabilitas tertinggi.
        2. Jelaskan perbandingan ketiga kategori.
        3. Jelaskan arti grafik secara sederhana.
        4. Gunakan bahasa Indonesia yang mudah dipahami mahasiswa.
        5. Jangan membuat diagnosis atau saran pengobatan.
        6. Tegaskan bahwa hasil ini hanya simulasi pembelajaran
           dan bukan diagnosis medis.

        Buat penjelasan dalam 2 sampai 3 paragraf.
        """

        # Mengirim permintaan ke Groq API
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "Kamu adalah asisten yang menjelaskan grafik hasil Machine Learning secara informatif dan berhati-hati."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 500
            },
            timeout=30
        )

        # Memeriksa respons Groq
        if response.status_code != 200:
            return Response({
                "success": False,
                "error": "Groq API gagal memproses permintaan.",
                "detail": response.text[:500]
            }, status=status.HTTP_502_BAD_GATEWAY)

        result = response.json()

        explanation = result["choices"][0]["message"]["content"]

        return Response({
            "success": True,
            "explanation": explanation
        })

    except requests.exceptions.Timeout:
        return Response({
            "success": False,
            "error": "Permintaan ke Groq API terlalu lama."
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)

    except requests.exceptions.RequestException:
        return Response({
            "success": False,
            "error": "Tidak dapat terhubung ke Groq API."
        }, status=status.HTTP_502_BAD_GATEWAY)

    except Exception as error:
        return Response({
            "success": False,
            "error": f"Terjadi kesalahan: {str(error)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET"])
@renderer_classes([JSONRenderer])
def get_riwayat(request):
    try:
        # Mengambil seluruh riwayat prediksi
        riwayat = RiwayatPrediksi.objects.all()

        data_riwayat = []

        for item in riwayat:
            data_riwayat.append({
                "id": item.id,
                "age": item.age,
                "gender": item.gender,
                "bmi": item.bmi,
                "prediction": item.prediction,
                "probabilities": {
                    "Low": item.low_probability,
                    "Moderate": item.moderate_probability,
                    "High": item.high_probability
                },
                "created_at": timezone.localtime(
                    item.created_at
                ).strftime("%d-%m-%Y %H:%M")
            })

        return Response({
            "success": True,
            "total": len(data_riwayat),
            "data": data_riwayat
        })

    except Exception as error:
        return Response({
            "success": False,
            "error": str(error)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


