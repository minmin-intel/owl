from camel.agents import ChatAgent
from camel.configs import DeepSeekConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.DEEPSEEK,
    model_type=ModelType.DEEPSEEK_CHAT,
    model_config_dict=DeepSeekConfig(temperature=0.2).as_dict(),
)

# Define system message
sys_msg = "You are a helpful assistant."

# Set agent
camel_agent = ChatAgent(system_message=sys_msg, model=model)

user_msg = """What are the three fun things to do in San Francisco?"""

# Get response information
response = camel_agent.step(user_msg)
print(response.msgs[0].content)




