# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

from dotenv import load_dotenv


import os

from camel.models import ModelFactory
from camel.logger import get_logger
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, TogetherAIConfig, DeepSeekConfig

from owl.utils import GAIABenchmark
from camel.logger import set_log_level

from tools.web_search import search_web_tool

import pathlib
import argparse

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")

logger = get_logger(__name__)

# Configuration
LEVEL = "all"#1
SAVE_RESULT = True
test_idx = [0]

def print_tools(tools):
    for tool in tools:
        print(tool.get_function_name())
        print(tool.get_function_description())
        print(tool.get_openai_function_schema())
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Run GAIA benchmark')
    parser.add_argument("--test_type", type=str, default="text", help="text or image")
    parser.add_argument("--model_provider", type=str, default="openai", help="Model provider")
    parser.add_argument("--save_to", type=str, default="result", help="Save results to")
    args = parser.parse_args()
    assert args.test_type in ["text", "image"], "Test type must be either text or image"

    """Main function to run the GAIA benchmark."""
    # Create cache directory
    cache_dir = "tmp/"
    os.makedirs(cache_dir, exist_ok=True)
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    if args.model_provider == "openai":
        platform = ModelPlatformType.OPENAI
        chat_model = ModelType.GPT_4O
        vlm = ModelType.GPT_4O
        model_config = ChatGPTConfig(temperature=0, top_p=1).as_dict()
        chat_platform = platform
        chat_model_config = model_config
        vlm_platform = platform
        vlm_model_config = model_config
    elif args.model_provider == "together":
        platform = ModelPlatformType.TOGETHER
        chat_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        # vlm = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo" 
        vlm="Qwen/Qwen2-VL-72B-Instruct"
        model_config = TogetherAIConfig(temperature=0.6).as_dict()
        chat_platform = platform
        chat_model_config = model_config
        vlm_platform = platform
        vlm_model_config = model_config
    elif args.model_provider == "hybrid":
        chat_platform = ModelPlatformType.DEEPSEEK
        chat_model = ModelType.DEEPSEEK_CHAT
        chat_model_config=DeepSeekConfig(temperature=0.6).as_dict()
        vlm_platform = ModelPlatformType.TOGETHER
        vlm = "Qwen/Qwen2-VL-72B-Instruct"
        # vlm = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        vlm_model_config = TogetherAIConfig(temperature=0.2).as_dict()
    else:
        raise ValueError(f"Model provider {args.model_provider} is not supported.")

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=chat_platform,
            model_type=chat_model,
            model_config_dict=chat_model_config,
        ),
        "assistant": ModelFactory.create(
            model_platform=chat_platform,
            model_type=chat_model,
            model_config_dict=chat_model_config,
        ),
        # "browsing": ModelFactory.create(
        #     model_platform=vlm_platform,
        #     model_type=vlm,
        #     model_config_dict=model_config,
        # ),
        # "planning": ModelFactory.create(
        #     model_platform=platform,
        #     model_type=chat_model,
        #     model_config_dict=model_config,
        # ),
        # "video": ModelFactory.create(
        #     model_platform=platform,
        #     model_type=vlm,
        #     model_config_dict=model_config,
        # ),
        "image": ModelFactory.create(
            model_platform=vlm_platform,
            model_type=vlm,
            model_config_dict=vlm_model_config,
        ),
    }

    # Configure toolkits
    if args.test_type == "image":
        tools = [
            # *BrowserToolkit(
            #     headless=False,  # Set to True for headless mode (e.g., on remote servers)
            #     web_agent_model=models["browsing"],
            #     planning_agent_model=models["planning"],
            # ).get_tools(), # two agents in the tool: planning agent(reasoning model) and web agent (VLM)
            # *VideoAnalysisToolkit(
            #     model=models["video"]
            # ).get_tools(),  # This requires OpenAI Key
            # *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
            *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(), #no model needed
            *ImageAnalysisToolkit(model=models["image"]).get_tools(), # need VLM
            # SearchToolkit().get_tools()[0], # wiki, google, ddg, tavily 0,2,3,5
            # SearchToolkit().get_tools()[2], # google
            # SearchToolkit().get_tools()[3], # ddg
            # SearchToolkit().get_tools()[5], #tavily
            search_web_tool,
            *ExcelToolkit().get_tools(), # no model needed
            *FileWriteToolkit(output_dir="./").get_tools(), # no model needed
        ]
    elif args.test_type == "text":
        tools = [
            *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(), #no model needed
            search_web_tool,
            *ExcelToolkit().get_tools(), # no model needed
            *FileWriteToolkit(output_dir="./").get_tools(), # no model needed
        ]
    else:
        raise ValueError(f"Test type {args.test_type} is not supported.")

    print_tools(tools)
    
    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Initialize benchmark
    benchmark = GAIABenchmark(data_dir="data/gaia", save_to=f"results/{args.save_to}.json", test_type=args.test_type)

    # Print benchmark information
    print(f"Number of validation examples: {len(benchmark.valid)}")
    print(f"Number of test examples: {len(benchmark.test)}")

    # Run benchmark
    result = benchmark.run(
        on="valid",
        level=LEVEL,
        idx=test_idx,
        save_result=SAVE_RESULT,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    # Output results
    logger.info(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.info(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
