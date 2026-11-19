from django.db import models

# Create your models here.

class RiwayatPrediksi(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    bmi = models.FloatField()

    prediction = models.CharField(max_length=20)

    low_probability = models.FloatField()
    moderate_probability = models.FloatField()
    high_probability = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.prediction} - {self.created_at}"
