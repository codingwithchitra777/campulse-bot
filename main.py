import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update

# Load configuration and environment variables
load_dotenv()

from app.core.logging_config import configure_logging
configure_logging()

from app.services.bot import CsxTradingBot
from app.core.config import settings

logger = logging.getLogger("campulse-bot")

async def main():
    logger.info("Initializing CamPulse Telegram Bot...")
    if not settings.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
        return
        
    bot_controller = CsxTradingBot()
    application = bot_controller.build_app()
    
    await application.initialize()
    await application.start()
    
    logger.info("Telegram Bot polling active...")
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        await application.stop()
        await application.shutdown()
        logger.info("Telegram Bot shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
