# ========================================
# 📦 IMPORTS & CONFIG
# ========================================
import argparse
import time
import csv
import sys
import os
import math
import subprocess
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from dronekit import connect, VehicleMode, Command
from pymavlink import mavutil
from ultralytics import YOLO
from sort import Sort
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
from itertools import cycle

warnings.filterwarnings('ignore')

# ========================================
# 📁 PATHS (Customize as needed)
# ========================================
BASE_DIR = "/home/nvidia/capstone"

CSV_PATH = f"{BASE_DIR}/Waypoints.csv"
GPS_LOG_FILE = f"{BASE_DIR}/dronepath.csv"
VISITED_POINTS_FILE = f"{BASE_DIR}/visited_points.csv"

MODEL_PATH = f"{BASE_DIR}/best.pt"
INPUT_VIDEO = f"{BASE_DIR}/video.mp4"
OUTPUT_VIDEO = f"{BASE_DIR}/detectedvideo.mp4"

OUTPUT_CSV = f"{BASE_DIR}/infected_crops.csv"
ZONE_SPRAY_PLAN = f"{BASE_DIR}/zone_spray_plan_drone_total.csv"

CIRCLE_ZONES_FILE = f"{BASE_DIR}/circle_zones_detection.csv"

GRAPH_OUTPUT_1 = f"{BASE_DIR}/drone_path_and_zones.jpg"
GRAPH_OUTPUT_2 = f"{BASE_DIR}/zones_vs_ground_truth.jpg"

GROUND_TRUTH_FILE = f"{BASE_DIR}/ground_truth.csv"
ORIGINAL_POINTS_FILE = f"{BASE_DIR}/original_points.csv"

WORKER_DRONE_IP = "192.168.1.50"
WORKER_DRONE_USER = "jetson"
REMOTE_DIR = "/home/jetson/capstone"

# Spray parameters
SPRAY_SPEED_ML_PER_SEC = 50  # ml/second
BASE_SPRAY_AMOUNT_PER_CROP = 5  # ml per diseased crop

# ========================================
# 🛫 VEHICLE CONTROL
# ========================================
def wait_ready(vehicle):
    """Wait for drone to become armable."""
    print("[i] Waiting for vehicle to be armable...")
    while not vehicle.is_armable:
        time.sleep(1)
    print("[i] Vehicle is armable.")

def arm_and_takeoff(vehicle, target_altitude):
    """Arm drone and take off to a given altitude."""
    print("[i] Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)

    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.5)

    print("[i] Taking off...")
    vehicle.simple_takeoff(target_altitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt or 0
        print(f"    Altitude: {alt:.1f} m")
        if alt >= target_altitude * 0.95:
            print("[i] Target altitude reached.")
            break
        time.sleep(1)

# ========================================
# 📍 MISSION SETUP
# ========================================
def read_waypoints(csv_path):
    """Load mission waypoints from CSV (latitude and longitude only)."""
    points = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row.get('latitude') or row.get('lat'))
            lon = float(row.get('longitude') or row.get('lon'))
            points.append((lat, lon))
    if not points:
        raise ValueError("No valid waypoints found in CSV.")
    return points

def upload_mission(vehicle, points, default_alt=50.0, add_rtl=False):
    """Upload waypoint mission to the drone."""
    print(f"[i] Uploading mission with {len(points)} waypoints...")
    cmds = vehicle.commands
    cmds.clear()

    for (lat, lon) in points:
        cmd = Command(0, 0, 0, 3, 16, 0, 1, 0, 0, 0, 0, lat, lon, default_alt)
        cmds.add(cmd)

    if add_rtl:
        rtl_cmd = Command(0, 0, 0, 3, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        cmds.add(rtl_cmd)

    cmds.upload()
    print("[i] Mission uploaded successfully.")

def start_mission(vehicle):
    """Start mission in AUTO mode."""
    print("[i] Starting mission...")
    vehicle.mode = VehicleMode("AUTO")
    while vehicle.mode.name != "AUTO":
        time.sleep(0.5)
    print("[i] Mission started in AUTO mode.")

# ========================================
# 🛰️ GPS LOGGING & CAMERA RECORDING
# ========================================
def log_gps_and_record(vehicle, num_waypoints):
    """Logs GPS coordinates and controls camera recording."""
    print("[i] Logging GPS and controlling camera...")
    
    with open(VISITED_POINTS_FILE, "w", newline="") as vf:
        vwriter = csv.writer(vf)
        vwriter.writerow(["latitude", "longitude"])
    

    with open(GPS_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude"])
        
        timestamp = 0
        camera_started = False
        current_waypoint = 0
        

        while vehicle.commands.next < 1:
            time.sleep(1)
        

        print("[i] Reached waypoint 0. Starting camera recording...")
        camera_started = True
        #
        
        while True:
            loc = vehicle.location.global_frame
            writer.writerow([timestamp, loc.lat, loc.lon])
            print(f"{timestamp}s | lat={loc.lat:.6f}, lon={loc.lon:.6f}")
            
            next_wp = vehicle.commands.next
            if next_wp > current_waypoint:
                current_waypoint = next_wp

                with open(VISITED_POINTS_FILE, "a", newline="") as vf:
                    vwriter = csv.writer(vf)
                    vwriter.writerow([loc.lat, loc.lon])
                print(f"[i] Reached waypoint {current_waypoint-1}: lat={loc.lat:.6f}, lon={loc.lon:.6f}")
            

            if next_wp >= num_waypoints:
                print("[i] Mission complete. Stopping camera...")
                #
                print("[i] GPS logging complete.")
                break
            
            time.sleep(1)
            timestamp += 1

# ========================================
# 🎯 DETECTION: YOLO
# ========================================
def detect_diseased_crops():
    """Detect diseased crops from video and log frame, ID, lat, lon."""
    model = YOLO(MODEL_PATH)
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    print("[📹] Waiting for video feed...")
    wait_count = 0
    max_wait = 30  
    while (not os.path.exists(INPUT_VIDEO) or os.path.getsize(INPUT_VIDEO) == 0) and wait_count < max_wait:
        print(f"[⏳] No video detected yet. Waiting... ({wait_count}/{max_wait}s)")
        time.sleep(1)
        wait_count += 1
    
    if wait_count >= max_wait:
        print("[!] Video file not found after 30 seconds. Exiting detection.")
        return
    
    print("[✅] Video feed detected. Starting detection...")
    

    gps_df = pd.read_csv(GPS_LOG_FILE)
    gps_times = gps_df['timestamp'].values
    gps_lats = gps_df['latitude'].values
    gps_lons = gps_df['longitude'].values

    def apply_gps_offset(base_lat, base_lon, centroid, track_id):
        max_jitter_m = 0.10  # 10 cm
        u_lat = (centroid[0] % 1000) / 1000.0
        u_lon = (centroid[1] % 1000) / 1000.0
        jitter_lat_m = u_lat * max_jitter_m
        jitter_lon_m = u_lon * max_jitter_m
        sign_lat = 1 if (track_id % 2 == 0) else -1
        sign_lon = 1 if (track_id % 3 == 0) else -1
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(base_lat)))
        offset_lat = sign_lat * jitter_lat_m * deg_per_m_lat
        offset_lon = sign_lon * jitter_lon_m * deg_per_m_lon
        return base_lat + offset_lat, base_lon + offset_lon

  
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracked_data = {}
    frame_idx = 0
    detection_records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO detection
        results = model.predict(frame, conf=0.35, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2])

        detections = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(detections)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            idx = min(frame_idx, len(gps_times) - 1)
            base_lat, base_lon = gps_lats[idx], gps_lons[idx]

   
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            lat, lon = apply_gps_offset(base_lat, base_lon, centroid, track_id)

            tracked_data.setdefault(track_id, {"lats": [], "lons": [], "frames": []})
            tracked_data[track_id]["lats"].append(lat)
            tracked_data[track_id]["lons"].append(lon)
            tracked_data[track_id]["frames"].append(frame_idx)


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 150), 3)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

        out.write(frame)

    cap.release()
    out.release()


    for tid, vals in tracked_data.items():
        for i, frame_num in enumerate(vals["frames"]):
            detection_records.append({
                "Frame": frame_num,
                "ID": tid,
                "latitude": vals["lats"][i],
                "longitude": vals["lons"][i]
            })
    
    pd.DataFrame(detection_records).to_csv(OUTPUT_CSV, index=False)
    print(f"[i] Detected crops saved → {OUTPUT_CSV}")
    print(f"[i] Detection video saved → {OUTPUT_VIDEO}")

# ========================================
# 🧭 ZONE CLUSTERING WITH DBSCAN
# ========================================
def cluster_zones_dbscan():
    print("\n[🌾] Running Dynamic DBSCAN Zone Clustering Analysis...")

    diseased_crops = pd.read_csv(OUTPUT_CSV)

    if 'ID' in diseased_crops.columns:
        diseased_crops = diseased_crops.drop_duplicates(subset=['ID'])
    diseased_crops = diseased_crops.dropna(subset=['latitude', 'longitude'])

    def latlon_to_meters(lat, lon, ref_lat, ref_lon):
        R = 6371000
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(ref_lat)
        ref_lon_rad = np.radians(ref_lon)
        x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        y = R * (lat_rad - ref_lat_rad)
        return x, y

    def meters_to_latlon(x, y, ref_lat, ref_lon):
        R = 6371000
        ref_lat_rad = np.radians(ref_lat)
        lat = ref_lat + np.degrees(y / R)
        lon = ref_lon + np.degrees(x / (R * np.cos(ref_lat_rad)))
        return lat, lon

    ref_lat = diseased_crops['latitude'].mean()
    ref_lon = diseased_crops['longitude'].mean()

    crop_x, crop_y = latlon_to_meters(
        diseased_crops['latitude'],
        diseased_crops['longitude'],
        ref_lat, ref_lon
    )

    X = np.column_stack([crop_x, crop_y])

    min_samples = max(10, int(0.02 * len(X)))

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    q1, q3 = np.percentile(k_distances, [25, 75])
    eps_value = q3 + 1.5 * (q3 - q1)

    db = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = db.fit_predict(X)

    diseased_crops['ZoneID'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[i] DBSCAN detected {n_clusters} dynamic zones (eps={round(eps_value,2)}, min_samples={min_samples})")

    valid_zones = [z for z in set(labels) if z != -1]
    zone_data = []

    for zone_id in valid_zones:
        zone_points = X[labels == zone_id]

        cx, cy = np.mean(zone_points, axis=0)

        radius_m = np.max(np.linalg.norm(zone_points - np.array([cx, cy]), axis=1))

        c_lat, c_lon = meters_to_latlon(cx, cy, ref_lat, ref_lon)
        crop_count = len(zone_points)
        spray_amount_ml = crop_count * BASE_SPRAY_AMOUNT_PER_CROP

        zone_data.append({
            'ZoneID': zone_id,
            'Centroid_Latitude': c_lat,
            'Centroid_Longitude': c_lon,
            'Diseased_Crops': crop_count,
            'Spray_Amount_ML': spray_amount_ml,
            'Zone_Radius_Meters': round(radius_m, 2)
        })

    zones_df = pd.DataFrame(zone_data)
    zones_df.to_csv(ZONE_SPRAY_PLAN, index=False)
    print(f"[i] Zone spray plan saved → {ZONE_SPRAY_PLAN}")

    circle_zones_data = []
    for _, zone in zones_df.iterrows():
        altitude_m = 30 + (zone['Zone_Radius_Meters'] * 2)
        duration_seconds = zone['Spray_Amount_ML'] / SPRAY_SPEED_ML_PER_SEC

        circle_zones_data.append({
            'ZoneID': zone['ZoneID'],
            'Latitude': zone['Centroid_Latitude'],
            'Longitude': zone['Centroid_Longitude'],
            'Altitude_Meters': round(altitude_m, 2),
            'Duration_Seconds': round(duration_seconds, 2),
            'Zone_Radius_Meters': zone['Zone_Radius_Meters']
        })

    circle_zones_df = pd.DataFrame(circle_zones_data)
    circle_zones_df.to_csv(CIRCLE_ZONES_FILE, index=False)
    print(f"[i] Circle zones detection saved → {CIRCLE_ZONES_FILE}")

    return zones_df, diseased_crops


# ========================================
# 📊 METRICS CALCULATION
# ========================================
def calculate_metrics():
    """Calculate MAE, IoU, Area Error, and Confusion Matrix metrics."""
    print("\n[📊] Calculating Metrics...")
    
    # 1. MAE - Mean Absolute Error between original and visited waypoints
    original_points = pd.read_csv(ORIGINAL_POINTS_FILE)
    visited_points = pd.read_csv(VISITED_POINTS_FILE)
    
    # Calculate MAE
    lat_mae = np.mean(np.abs(original_points['latitude'].values - visited_points['latitude'].values))
    lon_mae = np.mean(np.abs(original_points['longitude'].values - visited_points['longitude'].values))
    overall_mae = (lat_mae + lon_mae) / 2
    
    print(f"\n[MAE] Mean Absolute Error:")
    print(f"  Latitude MAE: {lat_mae:.8f}°")
    print(f"  Longitude MAE: {lon_mae:.8f}°")
    print(f"  Overall MAE: {overall_mae:.8f}°")
    
    # 2. IoU, Area Error, and Confusion Matrix
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    detected_crops = pd.read_csv(OUTPUT_CSV)
    
    # Convert to sets for IoU calculation
    gt_infected_ids = set(ground_truth[ground_truth['infected'] == 1]['id'].values)
    detected_ids = set(detected_crops['ID'].unique())
    
    # IoU calculation
    intersection = len(gt_infected_ids.intersection(detected_ids))
    union = len(gt_infected_ids.union(detected_ids))
    iou = intersection / union if union > 0 else 0
    
    print(f"\n[IoU] Intersection over Union: {iou:.4f}")
    
    # Area Error calculation (based on count as proxy for area)
    gt_area = len(gt_infected_ids)
    detected_area = len(detected_ids)
    area_error = abs(detected_area - gt_area) / gt_area if gt_area > 0 else 0
    
    print(f"[Area Error] {area_error:.4f} ({area_error*100:.2f}%)")
    
    # Confusion Matrix
    tp = len(gt_infected_ids.intersection(detected_ids)) 
    fp = len(detected_ids - gt_infected_ids)  
    fn = len(gt_infected_ids - detected_ids)  

    total_gt = len(ground_truth)
    tn = total_gt - (tp + fp + fn)
    tn = max(0, tn) 
    
    print(f"\n[Confusion Matrix]")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Negatives (FN): {fn}")
    
    # Calculate Recall and Accuracy
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\n[Recall] {recall:.4f} ({recall*100:.2f}%)")
    print(f"[Accuracy] {accuracy:.4f} ({accuracy*100:.2f}%)")

# ========================================
# 📊 VISUALIZATION
# ========================================
def visualize_results(zones_df, diseased_crops):
    """Create two visualization images."""
    print("\n[📊] Creating Visualizations...")
    
    drone_path = pd.read_csv(GPS_LOG_FILE)
    
    def latlon_to_xy(lat, lon, ref_lat, ref_lon):
        R = 6371000
        x = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
        y = R * np.radians(lat - ref_lat)
        return x, y
    
    ref_lat = np.mean(drone_path['latitude'])
    ref_lon = np.mean(drone_path['longitude'])
    
    # Convert coordinates
    drone_x, drone_y = latlon_to_xy(drone_path['latitude'], drone_path['longitude'], ref_lat, ref_lon)
    drone_x = np.array(drone_x)
    drone_y = np.array(drone_y)
    
    crops_x, crops_y = latlon_to_xy(diseased_crops['latitude'], diseased_crops['longitude'], ref_lat, ref_lon)
    zones_x, zones_y = latlon_to_xy(zones_df['Centroid_Latitude'], zones_df['Centroid_Longitude'], ref_lat, ref_lon)
    
    # === IMAGE 1: Drone Path and Zones ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Graph 1: Drone Path
    ax1.plot(drone_x, drone_y, 'b-', linewidth=2)
    ax1.plot(drone_x[0], drone_y[0], 'go', markersize=15, label='Start')
    ax1.plot(drone_x[-1], drone_y[-1], 'ro', markersize=15, label='End')
    ax1.set_title("Graph 1: Drone Flight Path", fontsize=15, fontweight='bold')
    ax1.set_xlabel("X Position (meters)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Y Position (meters)", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Graph 2: Diseased Crops Distribution with DBSCAN zones
    colors = [
        '#FF4444', '#44AA44', '#4444FF', '#FF8800',
        '#AA44AA', '#00AAAA', '#AA5500', '#5588FF', '#DD44AA'
    ]

    color_cycle = cycle(colors)

    ax2.scatter(crops_x, crops_y, c='red', s=40, alpha=0.6, edgecolor='darkred')

    for (_, zone), color in zip(zones_df.iterrows(), color_cycle):
        zx = zones_x.loc[zone.name]
        zy = zones_y.loc[zone.name]
        radius = zone['Zone_Radius_Meters']

        circle = Circle(
            (zx, zy),
            radius,
            fill=False,
            edgecolor=color,
            linewidth=3,
            linestyle='--'
        )
        ax2.add_patch(circle)

        ax2.text(
            zx,
            zy - radius - 1,
            f"Zone {int(zone['ZoneID'])}",
            ha='center',
            fontsize=11,
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc='white',
                ec=color,
                lw=2
            )
        )

    ax2.set_title("Graph 2: Diseased Crops Distribution", fontsize=15, fontweight='bold')
    ax2.set_xlabel("X Position (meters)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Y Position (meters)", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(GRAPH_OUTPUT_1, dpi=300, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"[i] Visualization 1 saved → {GRAPH_OUTPUT_1}")
    
    # === IMAGE 2: Zones vs Ground Truth Grid ===
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    

    gt_x, gt_y = latlon_to_xy(ground_truth['latitude'], ground_truth['longitude'], ref_lat, ref_lon)
    
    x_min, x_max = min(gt_x.min(), zones_x.min()) - 5, max(gt_x.max(), zones_x.max()) + 5
    y_min, y_max = min(gt_y.min(), zones_y.min()) - 5, max(gt_y.max(), zones_y.max()) + 5
    
    # Draw grid
    grid_size = 2  
    for x in np.arange(x_min, x_max, grid_size):
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
    for y in np.arange(y_min, y_max, grid_size):
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
    

    for _, gt_row in ground_truth.iterrows():
        gtx, gty = latlon_to_xy([gt_row['latitude']], [gt_row['longitude']], ref_lat, ref_lon)
        color = '#90EE90' if gt_row['infected'] == 0 else '#FFFF99' 
        rect = Rectangle((gtx[0]-1, gty[0]-1), 2, 2, 
                         facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.6)
        ax.add_patch(rect)
    
    # Overlay detected zones
    for idx, (_, zone) in enumerate(zones_df.iterrows()):
        zx = zones_x.iloc[idx]
        zy = zones_y.iloc[idx]
        radius = zone['Zone_Radius_Meters']
        
        circle = Circle((zx, zy), radius, fill=False,
                       color=colors[idx % len(colors)], linewidth=3, linestyle='--')
        ax.add_patch(circle)
        
        ax.text(zx, zy - radius - 1,
                f"Zone {int(zone['ZoneID'])}",
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', 
                         ec=colors[idx % len(colors)], lw=2))
    
    ax.set_title("Zones vs Ground Truth Grid", fontsize=15, fontweight='bold')
    ax.set_xlabel("X Position (meters)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Y Position (meters)", fontsize=13, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(GRAPH_OUTPUT_2, dpi=300, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"[i] Visualization 2 saved → {GRAPH_OUTPUT_2}")

# ========================================
# 🚁 SEND TO SECOND DRONE
# ========================================
def send_to_second_drone():
    print("\n[🚁] Sending zone CSV to worker drone via SCP...")

    if not os.path.exists(CIRCLE_ZONES_FILE):
        print(f"[!] CSV not found: {CIRCLE_ZONES_FILE}")
        return

    scp_cmd = [
        "scp",
        CIRCLE_ZONES_FILE,
        f"{WORKER_DRONE_USER}@{WORKER_DRONE_IP}:{REMOTE_DIR}/"
    ]

    try:
        subprocess.run(scp_cmd, check=True)
        print("[✓] CSV successfully sent to worker drone")
    except subprocess.CalledProcessError as e:
        print("[!] SCP transfer failed")
        print(e)

# ========================================
# 🧩 MAIN
# ========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", required=True, help="udp:127.0.0.1:14550")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--takeoff", type=float, default=50.0)
    parser.add_argument("--rtl", action="store_true")
    args = parser.parse_args()

    print(f"[i] Connecting to vehicle → {args.connect}")
    is_serial = args.connect.startswith("/") or args.connect.startswith("COM")
    vehicle = connect(args.connect, baud=args.baud if is_serial else None, wait_ready=True, timeout=120)

    try:
        # Phase 1: Flight Mission
        print("\n" + "="*50)
        print("PHASE 1: FLIGHT MISSION")
        print("="*50)
        wait_ready(vehicle)
        waypoints = read_waypoints(CSV_PATH)
        upload_mission(vehicle, waypoints, default_alt=50.0, add_rtl=args.rtl)
        arm_and_takeoff(vehicle, args.takeoff)
        start_mission(vehicle)
        log_gps_and_record(vehicle, len(waypoints))
        
        # Phase 2: Detection
        print("\n" + "="*50)
        print("PHASE 2: DISEASE DETECTION")
        print("="*50)
        detect_diseased_crops()
        
        # Phase 3: Zone Clustering
        print("\n" + "="*50)
        print("PHASE 3: ZONE CLUSTERING")
        print("="*50)
        zones_df, diseased_crops = cluster_zones_dbscan()
        
        # Phase 4: Metrics
        print("\n" + "="*50)
        print("PHASE 4: METRICS CALCULATION")
        print("="*50)
        calculate_metrics()
        
        # Phase 5: Visualization
        print("\n" + "="*50)
        print("PHASE 5: VISUALIZATION")
        print("="*50)
        visualize_results(zones_df, diseased_crops)
        
        # Phase 6: Send to Second Drone
        print("\n" + "="*50)
        print("PHASE 6: SEND TO SECOND DRONE")
        print("="*50)
        send_to_second_drone(args.second_drone)
        
        print("\n" + "="*50)
        print("✅ ALL PHASES COMPLETED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        print("\n[!] Mission interrupted.")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[i] Closing vehicle connection.")
        vehicle.close()


if __name__ == "__main__":
    main()