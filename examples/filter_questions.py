import json
import pandas as pd
import os

datapath = "data/gaia/2023/"

val_data = os.path.join(datapath, "validation/metadata.jsonl")
val_data = pd.read_json(val_data, lines=True)
print(val_data.columns)

# Filter questions
def filter_questions(data):
    # filter out questions that are about videos
    print(f"Before filtering: {data.shape[0]}")
    data = data[~data["Question"].str.contains("video", case=False)]
    print(f"After filtering: {data.shape[0]}")
    return data

def sample_per_level(data, n=5):
    # Sample n questions per level
    sampled_data = data.groupby("Level").apply(lambda x: x.sample(n=min(len(x), n), random_state=42)).reset_index(drop=True)
    print("Sampled data shape:", sampled_data.shape)
    return sampled_data

val_data = filter_questions(val_data)
val_data = sample_per_level(val_data)
output_file = os.path.join(datapath, "validation/metadata_filtered.jsonl")
val_data.to_json(output_file, orient="records", lines=True)

