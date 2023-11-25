"""
This module contains the telegram bot instance.
"""
from telegram.ext import ApplicationBuilder

from second_guessing_config import TELEGRAM_BOT_TOKEN

tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
