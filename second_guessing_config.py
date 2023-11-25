"""Configuration for the SecondGuessingGPT project."""
import os

from dotenv import load_dotenv

load_dotenv()

DJANGO_SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

DJANGO_DEBUG = (os.environ.get("DJANGO_DEBUG") or "false").lower() in ("true", "1", "yes", "y")
DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL") or "INFO"
SECOND_GUESSING_LOG_LEVEL = os.environ.get("SECOND_GUESSING_LOG_LEVEL") or "INFO"

DJANGO_HOST = os.environ["DJANGO_HOST"]
DJANGO_BASE_URL = f"https://{DJANGO_HOST}"
TELEGRAM_WEBHOOK_PATH = "telegram_webhook"

TELEGRAM_WEBHOOK_TOKEN = os.environ["TELEGRAM_WEBHOOK_TOKEN"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# TODO For debug mode support a list of bots in .env that are allowed to be activated via webhooks.
#  This is necessary because sometimes it might make sense to play with production db locally and we don't want to
#  suddenly repoint all the production bots to the local server.
