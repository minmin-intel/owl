import json
import pandas as pd
import os

datapath = "data/gaia/2023/"
# Filter questions
def filter_video_questions(data):
    # filter out questions that are about videos
    print(f"Before filtering: {data.shape[0]}")
    data = data[~data["Question"].str.contains("video", case=False)]
    data = data[~data["file_name"].str.contains("mp3|mp4", case=False)]
    print(f"After filtering: {data.shape[0]}")
    return data

def filter_browser_questions(data):
    # filter out questions that are about browsing
    print(f"Before filtering: {data.shape[0]}")
    retained = []
    for _, row in data.iterrows():
        if "browser" not in row["Annotator Metadata"]["Steps"].lower():
            retained.append(row["Question"])

    data = data[data["Question"].isin(retained)]
    # data = data[~data["Annotator Metadata"].str.contains("browser", case=False)]
    print(f"After filtering: {data.shape[0]}")
    return data

def get_questions_with_image(data):
    # file name ends with png, jpg, jpeg
    data_img = data[data["file_name"].str.contains("png|jpg|jpeg", case=True)]
    print(f"Questions with image: {data_img.shape[0]}")
    data_text = data[~data.index.isin(data_img.index)]
    print(f"Questions without image: {data_text.shape[0]}")
    return data_img, data_text
                

def sample_per_level(data, n=5):
    # Sample n questions per level
    sampled_data = data.groupby("Level").apply(lambda x: x.sample(n=min(len(x), n), random_state=42)).reset_index(drop=True)
    print("Sampled data shape:", sampled_data.shape)
    return sampled_data

def save(data, subset,output_name):
    output_file = os.path.join(datapath, f"{subset}/{output_name}.jsonl")
    data.to_json(output_file, orient="records", lines=True)
    data.to_csv(os.path.join(datapath, f"{subset}/{output_name}.csv"), index=False)


def process_data(subset = "validation"):
    val_data = os.path.join(datapath, f"{subset}/metadata.jsonl")
    val_data = pd.read_json(val_data, lines=True)
    print(val_data.columns)

    val_data = filter_video_questions(val_data)
    val_data = filter_browser_questions(val_data)

    save(val_data, subset, "metadata_filtered_no_video_browser")

    data_img, data_text = get_questions_with_image(val_data)

    save(data_img, subset,"metadata_filtered_image")
    save(data_text, subset,"metadata_filtered_text")

    if data_img.shape[0] >15:
        sampled_data_img = sample_per_level(data_img)
    else:
        sampled_data_img = data_img

    sampled_data_text = sample_per_level(data_text)

    save(sampled_data_img, subset,"metadata_filtered_image_sampled")
    save(sampled_data_text,subset, "metadata_filtered_text_sampled")

def get_tool_stats(subset="validation"):
    val_data = os.path.join(datapath, f"{subset}/metadata.jsonl")
    val_data = pd.read_json(val_data, lines=True)
    unique_tools = []
    for _, row in val_data.iterrows():
        tools = row["Annotator Metadata"]["Tools"]
        num_tools = row["Annotator Metadata"]["Number of tools"]
        if num_tools == 0:
            pass
        elif num_tools == 1:    
            unique_tools.append(tools)
        else:
            tool_list = tools.split("\n")
            tool_list = [tool.split(". ", 1)[-1] for tool in tool_list]
            unique_tools.extend(tool_list)

    unique_tools = list(set(unique_tools))
    for tool in unique_tools:
        if "None" in tool:
            unique_tools.remove(tool)
        if "No tool" in tool:
            unique_tools.remove(tool)

    print(f"Unique tools: {len(unique_tools)}")
    workdir = os.getenv("WORKDIR")
    dataset_path = os.path.join(workdir, "datasets/gaia/")
    output_path = os.path.join(dataset_path, f"{subset}_unique_tools.txt")
    with open(output_path, "w") as f:
        for tool in unique_tools:
            f.write(tool + "\n")

    web_browser_counter = 0
    search_engine_counter = 0
    visual_tool_counter = 0
    coding_tool_counter = 0
    document_tool_counter = 0
    audio_tool_counter = 0
    other_search_counter = 0
    api_counter = 0

    for tool in unique_tools:
        if "browser" in tool.lower():
            web_browser_counter += 1
        if "search" in tool.lower():
            search_engine_counter += 1
        if "image" in tool.lower() or "video" in tool.lower() or "color" in tool.lower() or "ocr" in tool.lower() or "vision" in tool.lower() or "gif" in tool.lower():
            visual_tool_counter += 1
        if "python" in tool.lower() or "calculator" in tool.lower() or "code" in tool.lower() or "script" in tool.lower() or "computer algebra" in tool.lower() or "programming" in tool.lower():
            coding_tool_counter += 1
        if "document" in tool.lower() or "pdf" in tool.lower() or "word" in tool.lower() or "excel" in tool.lower() or "csv" in tool.lower() or "spreadsheet" in tool.lower() or "editor" in tool.lower() or "powerpoint" in tool.lower() or "xls" in tool.lower() or "json" in tool.lower() or "file" in tool.lower():
            document_tool_counter += 1
        if "audio" in tool.lower() or "speech" in tool.lower():
            audio_tool_counter += 1
        if "wiki" in tool.lower() or "archive" in tool.lower() or "websites" in tool.lower():
            other_search_counter += 1
        if "youtube" in tool.lower() or "google map" in tool.lower() or "google translate" in tool.lower():
            api_counter += 1

    
    print(f"Web browser: {web_browser_counter} ({web_browser_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Search engine: {search_engine_counter} ({search_engine_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Visual tool: {visual_tool_counter} ({visual_tool_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Coding tool: {coding_tool_counter} ({coding_tool_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Document tool: {document_tool_counter} ({document_tool_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Audio tool: {audio_tool_counter} ({audio_tool_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"API tool: {api_counter} ({api_counter/(val_data.shape[0])*100:.2f}%)")
    print(f"Other search: {other_search_counter} ({other_search_counter/(val_data.shape[0])*100:.2f}%)")
    other_tools = len(unique_tools) - (web_browser_counter + search_engine_counter + visual_tool_counter + coding_tool_counter + document_tool_counter + other_search_counter + audio_tool_counter + api_counter)
    print(f"Other tools: {other_tools} ({other_tools/(val_data.shape[0])*100:.2f}%)")
    return unique_tools





if __name__ == "__main__":
    # process_data("validation")
    # process_data("test")
    tools = get_tool_stats("validation")



