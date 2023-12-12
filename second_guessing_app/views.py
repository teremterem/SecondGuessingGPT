"""
This module is used to define the views for the second_guessing_app.
"""
import asyncio
import json
import logging

from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, HttpResponseServerError

from second_guessing_app.agents.tg_update_handler import handle_telegram_update
from second_guessing_app.second_guessing_utils import csrf_exempt_async
from second_guessing_config import TELEGRAM_WEBHOOK_TOKEN

logger = logging.getLogger(__name__)


async def health_check(_: HttpRequest) -> HttpResponse:
    """Health check endpoint."""
    return HttpResponse("OK")


@csrf_exempt_async
async def telegram_webhook(request: HttpRequest) -> HttpResponse:
    """Telegram webhook endpoint."""
    try:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != TELEGRAM_WEBHOOK_TOKEN:
            return HttpResponseForbidden()

        async def _process_update() -> None:
            try:
                tg_update_dict = json.loads(request.body)
                await handle_telegram_update(tg_update_dict)
            except Exception as exc1:  # pylint: disable=broad-exception-caught
                logger.exception("ERROR WHILE PROCESSING UPDATE (level 1): %s\n\n%s\n\n", exc1, request.body)

        asyncio.create_task(_process_update())
        return HttpResponse("OK")

    except Exception as exc0:  # pylint: disable=broad-exception-caught
        logger.exception("ERROR WHILE PROCESSING UPDATE (level 0): %s", exc0)
        return HttpResponseServerError()
