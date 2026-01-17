"""
Notification Router
Endpoints for managing notifications and alerts.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

from utils.dependencies import verify_api_key
from services.telegram_bot import telegram_service, initialize_telegram

logger = logging.getLogger("crisisconnect.notifications")

router = APIRouter(
    prefix="/api/v1/notifications",
    tags=["Notifications"]
)

class TelegramConfig(BaseModel):
    token: str

class TestAlert(BaseModel):
    chat_id: str
    message: str

@router.post("/telegram/config", summary="Configure Telegram Bot")
async def configure_telegram(
    config: TelegramConfig,
    api_key: bool = Depends(verify_api_key)
):
    """
    Initialize the Telegram bot with a token.
    """
    try:
        initialize_telegram(config.token)
        return {"status": "success", "message": "Telegram bot initialized"}
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/telegram/test", summary="Send test alert")
async def send_test_alert(
    alert: TestAlert,
    api_key: bool = Depends(verify_api_key)
):
    """
    Send a test message to a specific Telegram chat.
    """
    if not telegram_service.bot:
        raise HTTPException(status_code=400, detail="Telegram bot not configured")
        
    success = await telegram_service.send_alert(alert.chat_id, alert.message)
    
    if success:
        return {"status": "success", "message": "Alert sent"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send alert")
