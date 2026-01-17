"""
Telegram Bot Service
Handles sending alerts to Telegram users.
"""

import logging
from typing import List, Optional
from telegram import Bot
from telegram.error import TelegramError
import asyncio
from datetime import datetime

logger = logging.getLogger("crisisconnect.telegram")

class TelegramBotService:
    """
    Service for sending notifications via Telegram.
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.bot = None
        if token:
            self.bot = Bot(token=token)
            logger.info("Telegram Bot initialized")
        else:
            logger.warning("Telegram Token not provided. Bot service disabled.")

    async def send_alert(self, chat_id: str, message: str) -> bool:
        """
        Write alert to a log file for testing purposes.
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] TO: {chat_id} | MSG: {message}\n"
            
            with open("alerts.log", "a", encoding="utf-8") as f:
                f.write(log_entry)
                
            logger.info(f"Logged alert to file for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")
            return False

    async def broadcast_alert(self, chat_ids: List[str], message: str):
        """
        Send an alert to multiple users.
        """
        if not self.bot:
            return
            
        tasks = [self.send_alert(chat_id, message) for chat_id in chat_ids]
        await asyncio.gather(*tasks)

# Singleton instance (will be initialized with token later)
telegram_service = TelegramBotService()

def initialize_telegram(token: str):
    """Initialize the global telegram service with a token."""
    global telegram_service
    telegram_service = TelegramBotService(token)
