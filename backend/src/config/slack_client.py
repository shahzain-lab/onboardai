from slack_sdk import WebClient
from config.env_config import config as env


slack_client = WebClient(token=env.SLACK_BOT_TOKEN)
