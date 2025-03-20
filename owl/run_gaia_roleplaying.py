from dotenv import load_dotenv
load_dotenv()

import os
from loguru import logger

from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    DocumentProcessingToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    WebToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, TogetherAIConfig

from utils import GAIABenchmark

import argparse


# Configuration
LEVEL = 1
SAVE_RESULT = True
test_idx = [0]


def main():
    parser = argparse.ArgumentParser(description='Run GAIA benchmark')
    parser.add_argument('--level', type=int, default=1, help='Level of the benchmark')
    parser.add_argument('--save_result', type=bool, default=True, help='Save the result')
    parser.add_argument("--model_provider", type=str, default="openai", help="Model provider")
    args = parser.parse_args()

    """Main function to run the GAIA benchmark."""
    # Create cache directory
    cache_dir = "tmp/"
    os.makedirs(cache_dir, exist_ok=True)

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



    # Create models for different components
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
        "web": ModelFactory.create(
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
        "search": ModelFactory.create(
            model_platform=platform,
            model_type=chat_model,
            model_config_dict=model_config,
        ),
    }
    
    # Configure toolkits
    tools = [
        *WebToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        # *VideoAnalysisToolkit(model=models["video"]).get_tools(),  # This requires OpenAI Key
        # *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        *SearchToolkit(model=models["search"]).get_tools(),
        *ExcelToolkit().get_tools(),
    ]
    
    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Initialize benchmark
    benchmark = GAIABenchmark(
        data_dir="data/gaia",
        save_to=f"results/result.json"
    )

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
    logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
