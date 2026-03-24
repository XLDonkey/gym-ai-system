#!/usr/bin/env python3
"""
XL Fitness AI Overseer — Sequence Extractor
Mac Mini training pipeline step 1.

Downloads video from Google Drive → runs YOLOv11-pose → extracts rep sequences
Usage: python3 extract_sequences.py <video_path> <label> <output_json>
       python3 extract_sequences.py /path/to/good_reps.mp4 good_rep output.json
"""

import cv2
import json
import math
import sys
import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO

# Load YOLO
print("Loading YOLOv11-pose...")
model = YOLO('yolo11n-pose.pt')
print("Ready.")

# COCO keypoint indices
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

def calc_angle(a, b, c):
    ab = [a[0]-b[0], a[1]-b[1]]
    cb = [c[0]-b[0], c[1]-b[1]]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.sqrt((ab[0]**2+ab[1]**2) * (cb[0]**2+cb[1]**2))
    return math.degrees(math.acos(max(-1, min(1, dot/mag)))) if mag > 0 else None

def get_best_elbow_angle(keypoints, conf_thresh=0.3):
    best_angle, best_conf = None, 0
    for s_i, e_i, w_i in [(KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST),
                           (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST)]:
        s, e, w = keypoints[s_i], keypoints[e_i], keypoints[w_i]
        if s[2] < conf_thresh or e[2] < conf_thresh or w[2] < conf_thresh:
            continue
        conf = (s[2]+e[2]+w[2])/3
        angle = calc_angle((s[0],s[1]), (e[0],e[1]), (w[0],w[1]))
        if angle and conf > best_conf:
            best_conf, best_angle = conf, angle
    return best_angle

def process_video(video_path, sample_every=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {os.path.basename(video_path)} — {total} frames @ {fps}fps")
    
    angles, times, fn = [], [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fn += 1
        if fn % sample_every: continue
        if fn % 300 == 0:
            print(f"  Frame {fn}/{total} ({fn/total*100:.0f}%)")
        
        results = model(frame, verbose=False, conf=0.3)
        angle = None
        if results and len(results[0].keypoints.data) > 0:
            # Pick largest person
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
                idx = areas.index(max(areas))
                kps = results[0].keypoints.data[idx].cpu().numpy()
                # Normalise
                kps[:,0] /= frame.shape[1]
                kps[:,1] /= frame.shape[0]
                angle = get_best_elbow_angle(kps)
        
        angles.append(angle)
        times.append(fn/fps)
    
    cap.release()
    print(f"  Extracted {len(angles)} readings")
    return angles, times, fps/sample_every

def extract_sequences(angles, times, eff_fps, label, min_rom=35):
    # Fill None values
    filled, last = [], 120
    for a in angles:
        if a is not None: last = a
        filled.append(last)
    
    # Smooth
    w = min(21, max(5, len(filled)//10*2+1))
    if w % 2 == 0: w += 1
    sm = savgol_filter(filled, w, 3) if len(filled) >= w else np.array(filled)
    
    # Find rep valleys and peaks
    valleys, _ = find_peaks(-sm, prominence=20, distance=int(eff_fps*0.8))
    peaks, _ = find_peaks(sm, prominence=20, distance=int(eff_fps*0.8))
    
    sequences = []
    for vi in valleys:
        pp = peaks[peaks < vi]
        np_ = peaks[peaks > vi]
        if not len(pp) or not len(np_): continue
        p1, p2 = pp[-1], np_[0]
        seg = sm[p1:p2+1]
        if len(seg) < 5: continue
        rom = float(np.max(seg) - np.min(seg))
        if rom < min_rom: continue
        dur = times[min(p2,len(times)-1)] - times[min(p1,len(times)-1)]
        if dur < 0.4 or dur > 15: continue
        sequences.append({
            'label': label,
            'min_angle': round(float(np.min(seg)),1),
            'rom': round(rom,1),
            'duration_s': round(dur,2),
            'avg_angle': round(float(np.mean(seg)),1),
            'std_angle': round(float(np.std(seg)),1),
            'source': os.path.basename(video_path)
        })
    
    print(f"  Extracted {len(sequences)} sequences (label: {label})")
    return sequences

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 extract_sequences.py <video> <label> <output.json>")
        print("Labels: good_rep, bad_rep, false_movement, user_seated_engaged, resting")
        sys.exit(1)
    
    video_path = sys.argv[1]
    label = sys.argv[2]
    output_path = sys.argv[3]
    
    angles, times, eff_fps = process_video(video_path)
    sequences = extract_sequences(angles, times, eff_fps, label)
    
    # Load existing sequences if output file exists
    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing sequences from {output_path}")
    
    all_sequences = existing + sequences
    with open(output_path, 'w') as f:
        json.dump(all_sequences, f, indent=2)
    
    counts = {}
    for s in all_sequences:
        counts[s['label']] = counts.get(s['label'],0)+1
    
    print(f"\n✅ Done: {len(all_sequences)} total sequences → {output_path}")
    print(f"Breakdown: {counts}")
