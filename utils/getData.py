import os
import pandas as pd
from utils.config import Config

def load_dataset_df(root_path=Config.DATA_ROOT):
    video_data = []
    categories = ['drowsiness', 'non-drowsiness']

    if not os.path.exists(root_path):
        print(f"ERROR: Path {root_path} not found.")
        return pd.DataFrame()

    for class_name in categories:
        class_path = os.path.join(root_path, class_name)
        if not os.path.exists(class_path): continue

        subjects = sorted([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])

        for subject in subjects:
            subject_path = os.path.join(class_path, subject)
            videos = [f for f in os.listdir(subject_path) if f.endswith('.mp4')]

            for video in videos:
                video_data.append({
                    'class': class_name,
                    'subject': subject,
                    'video': video,
                    'path': os.path.join(subject_path, video),
                    'label': 1 if class_name == 'drowsiness' else 0
                })

    return pd.DataFrame(video_data)