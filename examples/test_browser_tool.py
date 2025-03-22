
from camel.toolkits import BrowserToolkit
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, TogetherAIConfig
from camel.models import ModelFactory
from camel.logger import get_logger
from camel.logger import set_log_level

set_log_level(level="DEBUG")

logger = get_logger(__name__)

import argparse
parser = argparse.ArgumentParser(description='Run GAIA benchmark')
parser.add_argument("--model_provider", type=str, default="openai", help="Model provider")
args = parser.parse_args()

if args.model_provider == "openai":
    platform = ModelPlatformType.OPENAI
    chat_model = ModelType.GPT_4O
    vlm = ModelType.GPT_4O
    model_config = ChatGPTConfig(temperature=0, top_p=1).as_dict()
elif args.model_provider == "together":
    platform = ModelPlatformType.TOGETHER
    chat_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    vlm = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo" 
    model_config = TogetherAIConfig(temperature=0.2).as_dict()
else:
    raise ValueError(f"Model provider {args.model_provider} is not supported.")

models = {
        "user": ModelFactory.create(
            model_platform=platform,
            model_type=chat_model,
            model_config_dict=model_config,
        ),
        "assistant": ModelFactory.create(
            model_platform=platform,
            model_type=chat_model,
            model_config_dict=model_config,
        ),
        "browsing": ModelFactory.create(
            model_platform=platform,
            model_type=vlm,
            model_config_dict=model_config,
        ),
        "planning": ModelFactory.create(
            model_platform=platform,
            model_type=chat_model,
            model_config_dict=model_config,
        ),
        "video": ModelFactory.create(
            model_platform=platform,
            model_type=vlm,
            model_config_dict=model_config,
        ),
        "image": ModelFactory.create(
            model_platform=platform,
            model_type=vlm,
            model_config_dict=model_config,
        ),
    }

tools = [
        *BrowserToolkit(
            headless=True,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools()]

browse_url = tools[0]

# task = "tomorrow weather in San Francisco"
task = "What fruits are in the 2008 painting Embroidery from Uzbekistan"
start_url = "https://www.google.com"

resp = browse_url(task, start_url)

# browse_url tool not good even with openai GPT-4o  for this simple task