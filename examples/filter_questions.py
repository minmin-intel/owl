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


if __name__ == "__main__":
    # process_data("validation")
    process_data("test")



