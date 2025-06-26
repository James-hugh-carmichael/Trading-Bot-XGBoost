# config.py
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("APCA_LIVE_KEY_ID")
SECRET_KEY = os.getenv("APCA_LIVE_SECRET_KEY_ID")
BASE_URL = os.getenv("APCA_LIVE_BASE_URL")
