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

swarm-autonomous-drones-corn-disease-detection/
│
├── ai_detection/ # YOLOv8 training and inference
├── clustering/ # DBSCAN zone formation and severity analysis
├── drone_control/ # MAVLink & DroneKit scripts
│ ├── data_collection_drone/
│ └── worker_drone/
├── data/ # CSV outputs and processed data
├── reports/ # Metrics, graphs, and cost analysis
├── docs/ # Architecture and workflow diagrams
├── scripts/ # Automation scripts
├── requirements.txt
└── README.md


---

## 📊 Performance Metrics

- Detection Accuracy: **>92%**
- MAE (Data Collection Drone): **0.0967 m**
- MAE (Worker Drone): **0.087 m**
- IoU (Zone Detection vs Ground Truth): **0.914**
- Fungicide Reduction: **~72%**
- Cost Savings (1200 m² field): **₹2355**

---

## 🌱 Dataset Information

- Public dataset on Northern Corn Leaf Blight (NLB)
- ~3000 annotated images
- YOLO-compatible format
- Images and labels verified before training

---

## 🚁 Drone Capabilities

### Data Collection Drone
- Autonomous waypoint navigation
- Real-time video recording
- GPS & telemetry logging
- Onboard AI inference

### Worker Drone
- Autonomous navigation to zone centroids
- Precision spraying mechanism
- Spray duration based on disease severity
- Automatic return-to-launch (RTL)

---

## 📈 Key Outcomes

- Accurate disease detection and mapping
- Reliable zone-based clustering using DBSCAN
- Significant reduction in chemical usage
- Fully autonomous end-to-end operation
- Field-ready precision agriculture solution

---

## ⚠️ Disclaimer

This project is intended for **academic and research purposes only**.  
Real-world deployment requires regulatory approvals, safety testing, and compliance with local aviation and agricultural laws.

---

## 👨‍💻 Team

- **Likhith Diwakar**
- Karan H
- Kritik Agarwal
- Madhav H Nair

**Guide:** Prof. Ashok Patil  
**Institution:** PES University

---

## ⭐ Acknowledgements

- Open-source agricultural datasets
- Ultralytics YOLO
- MAVLink & DroneKit community



