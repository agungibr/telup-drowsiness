import os
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from utils.config import Config
from utils.preprocessing import apply_clahe, crop_face, extract_geometric_features, face_detection_model

def get_dino_model():
    model_name = "facebook/dinov2-large"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(Config.DEVICE)
    model.eval()
    return processor, model

def process_video(video_path, save_path, processor, model):
    if os.path.exists(save_path): return
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames, geo_feats = [], []
    last_geo = np.zeros(5, dtype=np.float32)
    last_geo[0] = 0.3 
    
    indices = np.linspace(0, total_frames-1, Config.SEQ_LEN, dtype=int) if total_frames > 0 else []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: 
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            geo_feats.append(last_geo)
            continue
            
        rgb = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
        det = face_detection_model.process(rgb)
        
        face = crop_face(rgb, det.detections[0]) if det.detections else cv2.resize(rgb, (224,224))
        geo = extract_geometric_features(rgb)
        
        if np.any(geo != 0): last_geo = geo
        else: geo = last_geo
            
        frames.append(cv2.resize(face, (224, 224)))
        geo_feats.append(geo)
        
    cap.release()
    
    vis_feats = []
    for i in range(0, len(frames), 8):
        batch = frames[i:i+8]
        if not batch: continue
        inputs = processor(images=batch, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            vis_feats.append(out)
            
    if vis_feats:
        final = np.concatenate([np.concatenate(vis_feats), np.array(geo_feats)], axis=1)
        np.save(save_path, final.astype(np.float32))

if __name__ == "__main__":
    from utils.getData import load_dataset_df
    from tqdm import tqdm
    
    os.makedirs(Config.FEATURE_DIR, exist_ok=True)
    df = load_dataset_df()
    proc, model = get_dino_model()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = f"{row['subject']}_{row['video'].replace('.mp4', '.npy')}"
        process_video(row['path'], os.path.join(Config.FEATURE_DIR, fname), proc, model)