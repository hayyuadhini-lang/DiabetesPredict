from django.urls import path
from django.urls import path
from . import api

urlpatterns = [
    path("predict/", api.predict_diabetes, name="predict_diabetes"),
    path("explain/", api.explain_graph, name="explain_graph"),
    path("riwayat/", api.get_riwayat, name="get_riwayat"),
]