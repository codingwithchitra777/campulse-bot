import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
import logfire

# Load configuration and environment variables
load_dotenv()

# Configure Logfire first thing so we get runtime logs in the Logfire console
logfire_token = os.getenv("LOGFIRE_TOKEN")
if logfire_token:
    try:
        logfire.configure(token=logfire_token, service_name="campulse-bot")
    except Exception as e:
        print(f"Logfire configuration failed: {e}", file=sys.stderr)

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
