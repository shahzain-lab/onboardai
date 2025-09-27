import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend root
BASE_DIR = Path(__file__).resolve().parents[2]  # /
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=DOTENV_PATH)

class Config:
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Slack
    SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
    SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
    SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

    # Google
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")

    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_NAME = os.getenv("DB_NAME", "onboardai")

    # JWT
    JWT_SECRET = os.getenv("JWT_SECRET", "changeme")

    # Vector DB
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Monitoring
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Export singleton
config = Config()
