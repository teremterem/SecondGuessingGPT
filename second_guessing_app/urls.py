"""
URLs for the second_guessing_app.
"""
from django.urls import path

from second_guessing_app import views
from second_guessing_config import TELEGRAM_WEBHOOK_PATH

urlpatterns = [
    path("h3a11h/", views.health_check, name="health_check"),
    path(f"{TELEGRAM_WEBHOOK_PATH}/", views.telegram_webhook, name="telegram_webhook"),
]
