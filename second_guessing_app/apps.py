"""
Defines the 'second_guessing_app' configuration with a default auto field and application name.
"""
import asyncio
import logging

from django.apps import AppConfig

from second_guessing_config import DJANGO_BASE_URL, TELEGRAM_WEBHOOK_PATH, TELEGRAM_WEBHOOK_TOKEN
from second_guessing_app.second_guessing_tg_bot import tg_app

logger = logging.getLogger(__name__)


class SecondGuessingAppConfig(AppConfig):
    """
    This class configures the 'second_guessing_app' Django application.
    It sets the default auto field and application name.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "second_guessing_app"

    def ready(self) -> None:
        """Initialize bots when the app is ready, by setting webhooks for the bots."""

        async def _ready():
            logger.info("SecondGuessingAppConfig.ready._ready() - entered")

            await tg_app.initialize()
            logger.info("TELEGRAM BOT INITIALIZED")

            try:
                webhook_url = f"{DJANGO_BASE_URL}/{TELEGRAM_WEBHOOK_PATH}/"
                await tg_app.bot.set_webhook(
                    url=webhook_url,
                    secret_token=TELEGRAM_WEBHOOK_TOKEN,
                )
                logger.info("TELEGRAM BOT WEBHOOK SET")
                # refresh the bot and set update the Telegram webhook
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("FAILED TO UPDATE TELEGRAM BOT WEBHOOK: %s", exc)
            logger.info("SecondGuessingAppConfig.ready._ready() - exited")

        asyncio.create_task(_ready())
