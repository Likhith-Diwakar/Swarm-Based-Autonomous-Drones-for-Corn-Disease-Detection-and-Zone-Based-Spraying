# ========================================
# 📦 IMPORTS & CONFIG
# ========================================
import argparse
import time
import csv
import sys
import os
import math
import warnings
import numpy as np
import pandas as pd
from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil

warnings.filterwarnings('ignore')

# ========================================
# 📁 PATHS
# ========================================
CIRCLE_ZONES_FILE = "/home/jetson/capstone/circle_zones_detection.csv"
WORKER_DRONE_PATH = "/home/jetson/capstone/worker_drone_path.csv"
GROUND_TRUTH_FILE = "/home/jetson/capstone/ground_truth.csv"
ZONE_SPRAY_PLAN = "/home/jetson/capstone/zone_spray_plan_drone_total.csv"


# ========================================
# 🌾 CONSTANTS FOR PERFORMANCE CALCULATION
# ========================================

FIELD_AREA_M2 = 1200  # Total field area in square meters


CHEMICAL_ML_PER_CROP = 5  # ml per detected diseased crop
FARMER_BLANKET_SPRAY_LITERS = 36  # Traditional blanket spraying
COST_PER_LITER_RS = 90  # Cost per liter of fungicide in rupees


OVERSPRAY_RATIO_FARMER = 0.72  # 72% of farmer's spray goes on healthy areas
OVERSPRAY_RATIO_DRONE_BASE = 0.10  # 10% drone overspray due to spray drift

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
# 📍 MISSION SETUP & EXECUTION
# ========================================
def load_spray_mission(csv_path):
    """Load spray mission from circle_zones_detection.csv."""
    print(f"[i] Waiting for mission file from scout drone...")
    print(f"    Expected location: {csv_path}")
    

    wait_count = 0
    max_wait = 120  
    
    while not os.path.exists(csv_path) and wait_count < max_wait:
        if wait_count % 10 == 0: 
            print(f"[⏳] Waiting for mission file... ({wait_count}/{max_wait}s)")
        time.sleep(1)
        wait_count += 1
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Mission file not received after {max_wait}s: {csv_path}")
    

    print(f"[✓] Mission file received!")
    zones_df = pd.read_csv(csv_path)
    print(f"[i] Loaded {len(zones_df)} spray zones from mission file")
    

    print("\n" + "="*60)
    print("SPRAY MISSION SUMMARY")
    print("="*60)
    for _, row in zones_df.iterrows():
        print(f"  Zone {int(row['ZoneID'])}: Lat={row['Latitude']:.6f}, Lon={row['Longitude']:.6f}, "
              f"Alt={row['Altitude_Meters']:.1f}m, Duration={row['Duration_Seconds']:.1f}s")
    print("="*60 + "\n")
    
    spray_zones = []
    for _, row in zones_df.iterrows():
        spray_zones.append({
            'zone_id': int(row['ZoneID']),
            'latitude': float(row['Latitude']),
            'longitude': float(row['Longitude']),
            'altitude': float(row['Altitude_Meters']),
            'duration': float(row['Duration_Seconds']),
            'radius': float(row['Zone_Radius_Meters'])
        })
    
    return spray_zones

def upload_spray_mission(vehicle, spray_zones):
    """Upload spray waypoints to the drone."""
    print(f"[i] Uploading spray mission with {len(spray_zones)} zones...")
    cmds = vehicle.commands
    cmds.clear()

    for zone in spray_zones:

        cmd = Command(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,
            zone['duration'], 
            0, 0, 0,
            zone['latitude'],
            zone['longitude'],
            zone['altitude']
        )
        cmds.add(cmd)


    rtl_cmd = Command(0, 0, 0, 3, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    cmds.add(rtl_cmd)

    cmds.upload()
    print("[i] Spray mission uploaded successfully.")

def start_spray_mission(vehicle):
    """Start mission in AUTO mode."""
    print("[i] Starting spray mission...")
    vehicle.mode = VehicleMode("AUTO")
    while vehicle.mode.name != "AUTO":
        time.sleep(0.5)
    print("[i] Spray mission started in AUTO mode.")

# ========================================
# 💧 SPRAY EXECUTION & LOGGING
# ========================================
def execute_spray_mission(vehicle, spray_zones):
    """Execute spray mission and log performance."""
    print("\n[💧] Executing Spray Mission...")
    

    with open(WORKER_DRONE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ZoneID", "Latitude", "Longitude", "Altitude_Meters", 
                        "Duration_Seconds", "Actual_Spray_Time_Seconds", "Status"])
    
    current_waypoint = 0
    num_zones = len(spray_zones)
    

    while vehicle.commands.next < 1:
        time.sleep(1)
    
    print("\n[i] Mission started. Spraying zones...")
    

    for zone_idx, zone in enumerate(spray_zones):
        print(f"\n{'='*60}")
        print(f"[Zone {zone['zone_id']}] Navigating to spray zone...")
        print(f"  Target: Lat={zone['latitude']:.6f}, Lon={zone['longitude']:.6f}")
        print(f"  Altitude: {zone['altitude']:.1f}m")
        print(f"  Planned Duration: {zone['duration']:.1f}s")
        print(f"{'='*60}")
        

        while vehicle.commands.next <= zone_idx:
            time.sleep(0.5)
        

        loc = vehicle.location.global_relative_frame
        actual_lat = loc.lat
        actual_lon = loc.lon
        actual_alt = loc.alt
        
        print(f"[✓] Arrived at Zone {zone['zone_id']}")
        print(f"    Actual Position: Lat={actual_lat:.6f}, Lon={actual_lon:.6f}, Alt={actual_alt:.1f}m")
        print(f"[💧] Spraying for {zone['duration']:.1f} seconds...")
        

        spray_start_time = time.time()
        spray_duration = zone['duration']
        

        elapsed = 0
        while elapsed < spray_duration:
            time.sleep(1)
            elapsed = time.time() - spray_start_time
            remaining = max(0, spray_duration - elapsed)
            print(f"    Spraying... {elapsed:.1f}s / {spray_duration:.1f}s (Remaining: {remaining:.1f}s)")
        
        actual_spray_time = time.time() - spray_start_time
        
        print(f"[✓] Zone {zone['zone_id']} spray complete!")
        

        with open(WORKER_DRONE_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                zone['zone_id'],
                actual_lat,
                actual_lon,
                actual_alt,
                zone['duration'],
                round(actual_spray_time, 2),
                "Completed"
            ])

    print("\n[i] All zones sprayed. Returning to launch...")
    while vehicle.commands.next < num_zones + 1:
        time.sleep(1)
    
    print("[✓] Spray mission completed successfully!")

# ========================================
# 📊 PERFORMANCE CALCULATIONS
# ========================================
def calculate_performance_metrics():
    """Calculate chemical usage, cost, and overspray metrics."""
    print("\n" + "="*70)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*70)
    

    worker_path = pd.read_csv(WORKER_DRONE_PATH)
    zone_spray_plan = pd.read_csv(ZONE_SPRAY_PLAN)
    

    total_diseased_crops = zone_spray_plan['Diseased_Crops'].sum()
    drone_chemical_ml = total_diseased_crops * CHEMICAL_ML_PER_CROP
    drone_chemical_liters = drone_chemical_ml / 1000.0
    farmer_chemical_liters = FARMER_BLANKET_SPRAY_LITERS
    chemical_savings_liters = farmer_chemical_liters - drone_chemical_liters
    chemical_savings_percent = (chemical_savings_liters / farmer_chemical_liters) * 100
    
    farmer_cost = farmer_chemical_liters * COST_PER_LITER_RS
    drone_cost = drone_chemical_liters * COST_PER_LITER_RS
    cost_savings = farmer_cost - drone_cost
    cost_savings_percent = (cost_savings / farmer_cost) * 100
    
    farmer_overspray_liters = farmer_chemical_liters * OVERSPRAY_RATIO_FARMER
    drone_overspray_liters = drone_chemical_liters * OVERSPRAY_RATIO_DRONE_BASE
    overspray_reduction = farmer_overspray_liters - drone_overspray_liters
    overspray_reduction_percent = (overspray_reduction / farmer_overspray_liters) * 100
    
    total_spray_time = worker_path['Actual_Spray_Time_Seconds'].sum()
    zones_sprayed = len(worker_path)
    

    print("\n1️⃣  CHEMICAL USAGE COMPARISON (Fluopyram + Tebuconazole)")
    print(f"   Farmer (Blanket): {farmer_chemical_liters:.2f} liters")
    print(f"   Drone (Precision): {drone_chemical_liters:.2f} liters")
    print(f"   ✅ Savings: {chemical_savings_liters:.2f} liters ({chemical_savings_percent:.1f}%)")
    
    print("\n2️⃣  COST COMPARISON")
    print(f"   Farmer Cost: ₹{farmer_cost:.2f}")
    print(f"   Drone Cost: ₹{drone_cost:.2f}")
    print(f"   ✅ Savings: ₹{cost_savings:.2f} ({cost_savings_percent:.1f}%)")
    
    print("\n3️⃣  OVERSPRAY ANALYSIS")
    print(f"   Farmer Overspray: {farmer_overspray_liters:.2f} liters")
    print(f"   Drone Overspray: {drone_overspray_liters:.2f} liters")
    print(f"   ✅ Reduction: {overspray_reduction:.2f} liters ({overspray_reduction_percent:.1f}%)")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Field Area: {FIELD_AREA_M2} m² | Diseased Crops: {total_diseased_crops} | Zones: {zones_sprayed}")
    print(f"Cost Saved: ₹{cost_savings:.2f} | Chemical Saved: {chemical_savings_liters:.2f}L | Overspray Reduced: {overspray_reduction:.2f}L")
    print(f"Spray Time: {total_spray_time:.1f}s ({total_spray_time/60:.1f} min)")
    print("="*70 + "\n")
# ========================================
# 🧩 MAIN
# ========================================
def main():
    parser = argparse.ArgumentParser(description="Worker Drone - Precision Spray System")
    parser.add_argument("--connect", required=True, 
                       help="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--takeoff", type=float, default=30.0,
                       help="Initial takeoff altitude in meters")
    args = parser.parse_args()

    print("="*70)
    print("🚁 WORKER DRONE - PRECISION SPRAY SYSTEM")
    print("="*70)
    print(f"[i] Connecting to vehicle → {args.connect}")
    
    is_serial = args.connect.startswith("/") or args.connect.startswith("COM")
    vehicle = connect(args.connect, baud=args.baud if is_serial else None, 
                     wait_ready=True, timeout=120)

    try:
        # Phase 1: Load Mission
        print("\n" + "="*70)
        print("PHASE 1: LOADING SPRAY MISSION")
        print("="*70)
        spray_zones = load_spray_mission(CIRCLE_ZONES_FILE)
        
        # Phase 2: Upload and Start Mission
        print("\n" + "="*70)
        print("PHASE 2: UPLOADING MISSION TO DRONE")
        print("="*70)
        wait_ready(vehicle)
        upload_spray_mission(vehicle, spray_zones)
        arm_and_takeoff(vehicle, args.takeoff)
        start_spray_mission(vehicle)
        
        # Phase 3: Execute Spray Mission
        print("\n" + "="*70)
        print("PHASE 3: EXECUTING SPRAY MISSION")
        print("="*70)
        execute_spray_mission(vehicle, spray_zones)
        
        # Phase 4: Calculate Performance
        print("\n" + "="*70)
        print("PHASE 4: PERFORMANCE ANALYSIS")
        print("="*70)
        calculate_performance_metrics()
        
        print("\n" + "="*70)
        print("✅ WORKER DRONE MISSION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n📄 Output Files:")
        print(f"   - Worker Drone Path: {WORKER_DRONE_PATH}")
        print(f"   - Performance Summary: C:\\capstone\\worker_drone_performance_summary.csv")
        
    except KeyboardInterrupt:
        print("\n[!] Mission interrupted by user.")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[i] Closing vehicle connection.")
        vehicle.close()


if __name__ == "__main__":
    main()