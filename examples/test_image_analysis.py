from camel.toolkits import ImageAnalysisToolkit
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, TogetherAIConfig
from camel.models import ModelFactory
from camel.logger import get_logger
from camel.logger import set_log_level

set_log_level(level="DEBUG")

logger = get_logger(__name__)

platform = ModelPlatformType.TOGETHER
chat_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
vlm = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo" 
model_config = TogetherAIConfig(temperature=0.2).as_dict()

model = ModelFactory.create(
            model_platform=platform,
            model_type=vlm,
            model_config_dict=model_config,
        )
tools = [*ImageAnalysisToolkit(model=model).get_tools()]

# print(tools)
imageUrl = "https://napkinsdev.s3.us-east-1.amazonaws.com/next-s3-uploads/d96a3145-472d-423a-8b79-bca3ad7978dd/trello-board.png"
imageUrl = "data/gaia/2023/validation/df6561b2-7ee5-4540-baab-5095f742716a.png"
image_to_text = tools[0]
resp = image_to_text(imageUrl)
print(resp)

