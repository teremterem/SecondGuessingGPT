"""
Defines the 'second_guessing_app' configuration with a default auto field and application name.
"""
from django.apps import AppConfig


class SecondGuessingAppConfig(AppConfig):
    """
    This class configures the 'second_guessing_app' Django application.
    It sets the default auto field and application name.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "second_guessing_app"
