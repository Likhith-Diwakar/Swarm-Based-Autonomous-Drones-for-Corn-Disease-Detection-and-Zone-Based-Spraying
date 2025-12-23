# Swarm-Based-Autonomous-Drones-for-Corn-Disease-Detection-and-Zone-Based-Spraying
# Swarm-Based Autonomous Drones for Corn Disease Detection and Zone-Based Spraying

An end-to-end autonomous precision agriculture system that uses a swarm of drones, AI-based disease detection, and zone-based fungicide spraying to accurately detect and control Northern Corn Leaf Blight (NCLB) while minimizing chemical usage and operational costs.

---

## 📌 Project Overview

Traditional agricultural practices rely on blanket spraying of fungicides, leading to excessive chemical usage, higher costs, and environmental damage.  
This project proposes a **swarm-based autonomous drone solution** that detects diseased crops using computer vision and performs **targeted, zone-based spraying** only where infection is present.

The system consists of:
- A **Data Collection Drone** for detection and analysis
- A **Worker Drone** for precision fungicide spraying

The entire pipeline operates autonomously with minimal human intervention.

---

## 🎯 Objectives

- Early detection of Northern Corn Leaf Blight in corn fields
- Accurate mapping of infected crop locations using GPS
- Formation of biologically realistic disease zones
- Precision spraying to reduce fungicide usage
- Autonomous multi-drone coordination
- Generation of actionable reports and metrics

---

## 🧠 System Architecture

1. Data collection drone autonomously flies predefined waypoints
2. Onboard AI detects diseased crops from video feed
3. Detections are mapped with GPS coordinates
4. DBSCAN clustering creates realistic infection zones
5. Zone centroids and severity are calculated
6. Worker drone autonomously visits each zone
7. Fungicide is sprayed based on zone severity
8. System logs accuracy, cost, and chemical usage metrics

---

## ⚙️ Technologies Used

- Python
- YOLOv8 (Object Detection)
- OpenCV
- SORT (Object Tracking)
- DBSCAN (Clustering)
- Pandas & NumPy
- MAVLink
- DroneKit
- Pixhawk Flight Controller
- Intel NUC
- QGroundControl

---

## 📂 Repository Structure


