# 🚛 Freight AI: Automated Freight Compliance & Cargo Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Seg-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-USDOT%20%2F%20FHWA-navy.svg)]()

**Automated Freight Compliance & Cargo Instance Segmentation** is an intelligent computer vision and multimodal vision-language pipeline developed for roadside freight enforcement, weigh-in-motion (WIM) gate analysis, and Over-Dimensional (OS/OW) permit compliance[cite: 2, 3]. 

Engineered under the **U.S. Department of Transportation (USDOT) Federal Highway Administration (FHWA) DOTSI Program**, this system combines real-time **YOLOv8-Seg** polygon extraction with zero-shot **Multimodal LLMs (Gemini)** and discrete geometric spatial surface area integration[cite: 2, 3].

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8--Seg-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-CC0--1.0%20%2F%20USDOT-navy.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## 📌 Outline / Table of Contents

- [Project Description](#-project-description)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Utility Functions & System Architecture](#-utility-functions--system-architecture)
- [Additional Notes & Context on Previous Work](#-additional-notes--context-on-previous-work)
- [Version History and Retention](#-version-history-and-retention)
- [License](#-license)
- [Contributions](#-contributions)
- [Contact Information](#-contact-information)
- [Acknowledgements & Contributors](#-acknowledgements--contributors)

---

## 🏛️ Project Description

### Title
**Automated Freight Compliance & Cargo Instance Segmentation (Freight AI)**

### Purpose and Goals of the Project
Commercial freight enforcement is critical to protecting highway infrastructure from pavement fatigue, accelerated structural degradation, and bridge deterioration caused by overweight vehicles. State Departments of Transportation (DOTs) enforce strict regulations regarding commercial vehicle load divisibility:
* **Divisible Loads:** Freight consisting of multi-unit, granular, or separable cargo (e.g., aggregate, soil, gravel, timber, boxed goods) subject to standard statutory gross vehicle and axle weight limits.
* **Non-Divisible Loads:** Monolithic structural elements, heavy industrial machinery, or single integrated assemblies eligible for state Over-Dimensional / Overweight (OS/OW) permits.

Traditional physical inspections at roadside weigh stations create severe highway corridor bottlenecks and operational delays. **Freight AI** addresses these challenges by developing an automated data collection, computer vision, and multimodal AI modeling pipeline that performs real-time cargo instance segmentation, spatial geometry surface area integration, and statutory compliance classification directly from roadside camera feeds (e.g., I-295 Southbound corridor).

## 🏛️ Executive Overview

Overweight commercial freight poses significant risks to highway infrastructure, accelerating pavement fatigue and bridge deterioration[cite: 2]. State DOTs enforce strict load divisibility rules:
* **Divisible Loads:** Granular, multi-unit, or loose bulk freight (e.g., aggregate, soil, timber, gravel) subject to legal axle limits[cite: 2].
* **Non-Divisible Loads:** Monolithic equipment, machinery, or structural assemblies eligible for Over-Dimensional (OS/OW) state permits[cite: 2].

Manual inspections at physical weigh stations cause severe corridor congestion[cite: 2]. **Freight AI** automates cargo isolation and compliance classification directly from highway camera feeds (e.g., I-295 SB corridor)[cite: 2, 3]:
* **Pixel-Level Cargo Isolation:** Traces exact irregular payload contours while ignoring vehicle tractor cabs, tires, and surrounding passenger cars[cite: 2, 3].
* **Spatial Surface Area Computation:** Computes 2D payload surface area from predicted polygon coordinates using Gauss's Shoelace Integration[cite: 2].
* **Divisibility Classification:** Categorizes loads into `divisible`, `likely_divisible`, `non_divisible`, or `unknown` using a 5-tier rule hierarchy[cite: 2, 3].

---

## 📸 Commercial Freight Visual Gallery & Body Types

To ensure accurate segmentation across diverse commercial fleets, the dataset is categorized into core body types[cite: 3]. The system applies strict **Payload Isolation Rules**, ignoring steer axles, tractor cabs, and surrounding passenger cars while tracing the cargo compartment[cite: 3].

---

### 1. End-Dumps & Vocational Dump Trucks
![End-Dump Soil Payload Segmentation](docs/images/dump_truck_segmentation.png)

* **Vehicle Characteristics:** End-dumps and vocational haulers carrying mounded soil, loose gravel, aggregate, or tarped bulk material[cite: 3].
* **Segmentation Protocol:** Polygon boundaries trace the dump bed box, cab guard overhang, and the irregular contour of mounded aggregate or protective tarp above the rim, stopping immediately above the rear dual tandem axles[cite: 3].
* **Compliance Classification:** Classified as `divisible` under visual payload rules, as loose aggregate can be safely offloaded to meet legal axle limits[cite: 3].

---

### 2. Flatbeds & Over-Dimensional (OS/OW) Heavy Machinery
![Flatbed Heavy Machinery Segmentation](docs/images/flatbed_excavator_segmentation.png)

* **Vehicle Characteristics:** Flatbed and lowboy trailers carrying heavy construction machinery, industrial excavators, steel beams, or timber stacks[cite: 3].
* **Segmentation Protocol:** Polygon vertices outline the explicit outer perimeter of the monolithic machinery or strapped payload, isolating the payload deck from the tractor cab[cite: 3].
* **Compliance Classification:** Classified as `non_divisible` when carrying single integrated machinery items, triggering automated permit validation for state Over-Dimensional (OS/OW) regulations[cite: 2, 3].

---

### 3. Specialized Vocational Equipment & Concrete Mixers
![Concrete Mixer Segmentation](docs/images/concrete_mixer_segmentation.png)

* **Vehicle Characteristics:** Specialized vocational vehicles, including rotating drum concrete mixers, vacuum excavators, and liquid tankers[cite: 3].
* **Segmentation Protocol:** Polygon vertices enclose the tank shell, rotating drum assembly, side toolboxes, and rear discharge chutes behind the main cab line[cite: 3].
* **Compliance Classification:** Concrete mixers and liquid tankers are governed by state commodity rules and classified as `divisible` or `non_divisible` based on jurisdiction (e.g., fluid milk exceptions vs. slurry mixers)[cite: 3].

---

### 4. Dry Vans, Reefers & Enclosed Freight Box Trailers
![Dry Van Enclosed Trailer Segmentation](docs/images/dry_van_segmentation.png)

* **Vehicle Characteristics:** Enclosed commercial cargo box bodies, refrigerated vans (reefers), and intermodal container chassis[cite: 3].
* **Segmentation Protocol:** Traces the rectangular box perimeter along the top edge, front bulkhead, and bottom trailer skirt, stopping directly above the rear wheel tandems[cite: 3].
* **Compliance Classification:** Generally classified as `divisible` or `likely_divisible` depending on door visibility, tarp securement, and cargo manifest data[cite: 3].

---

### 5. Multimodal Vision-Language High-Density Coordinate Grid Overlay
![High-Density Grid Overlay Inspection](docs/images/grid_overlay_inspection.png)

* **Coordinate Resolution:** Overlay grid spanning **0 to 1800 on the X-axis** and **0 to 1000 on the Y-axis**[cite: 3].
* **Spatial Prompt Mapping:** Enables Gemini Multimodal LLMs to map precise vertex pairs (`load_polygon: [[x1, y1], [x2, y2], ...]`) directly from visual grid intersections[cite: 3].
* **Inspection HUD Overlay:** Generates a real-time inspection HUD displaying File ID, Classification, Confidence Score, Jurisdiction, Cargo Type, and Applied Permitting Rule[cite: 3].

---

## ✨ Key Features

* **YOLOv8-Seg Real-Time Edge Screening:** High-speed instance segmentation running at **12.4 ms per image** for roadside bypass gates at 55 mph[cite: 2].
* **Multimodal LLM Reasoning (Gemini):** High-precision zero-shot vision-language evaluation with **100% JSON schema compliance**[cite: 2].
* **Target Payload Isolation:** Strict annotation rules that trace exclusively cargo payload perimeters across 6 trailer body types (End-dumps, Refuse, Tankers, Flatbeds, Auto Carriers, Dry Vans)[cite: 3].
* **Shoelace Surface Area Integration:** Discrete geometry pipeline calculating exact pixel surface area directly from predicted polygon vertices[cite: 2]:
  $$\text{Area} = 0.5 \cdot \left\vert{} \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right\vert{}$$
* **Dynamic Data Quality Guardrails:** Automated dataset validator enforcing a strict ambiguity ceiling to prevent class imbalance collapse[cite: 2]:
  $$\text{Ratio}_{\text{likely\_divisible}} \le 0.15$$

---

## ⚙️ System Architecture & Hybrid Deployment

Freight AI utilizes a **Hybrid Deployment Model** to balance low latency with deep contextual reasoning[cite: 2]:

```text
                          ┌───────────────────────────────┐
                          │ Highway Monitoring Camera     │
                          │ (I-295 SB Pass / Drive Feed)  │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │  Data Filtering & Pre-Proc  │
                           └──────────────┬──────────────┘
                                          │
                ┌─────────────────────────┴─────────────────────────┐
                │                                                   │
                ▼                                                   ▼
 ┌─────────────────────────────┐                     ┌─────────────────────────────┐
 │ Real-Time Edge Deployment   │                     │  Cloud Audit & Retraining   │
 │ (YOLOv8-Seg Instance Model) │                     │   (Gemini Multimodal LLM)   │
 ├─────────────────────────────┤                     ├─────────────────────────────┤
 │ • Latency: 12.4 ms          │                     │ • Latency: ~1,400 ms        │
 │ • Purpose: 55 mph Screening │                     │ • Purpose: Permit Audit /   │
 │ • Output: Pixel Polygon     │                     │   Automated Auto-Labeling   │
 └─────────────────────────────┘                     └─────────────────────────────┘

### Purpose of the Source Code & Relation to Project Goals
This repository houses the core software engine, data preprocessing scripts, deep learning segmentation pipelines, vision-language prompt frameworks, and validation tools required to automate freight compliance. Specifically, the codebase executes:
1. **Target Payload Isolation:** Isolates cargo payloads while explicitly ignoring truck tractor cabs, steer axles, tires, undercarriages, and surrounding traffic.
2. **Polygon Coordinate Extraction & Discrete Surface Geometry:** Traces irregular payload perimeters and computes exact 2D pixel surface areas using Gauss's Shoelace Area Integration.
3. **Multimodal Compliance Classification:** Employs fine-tuned Vision-Language Models (Gemini) with structured JSON schemas to enforce state-specific commodity and permitting rules.

### Intended Audience
This project is engineered for state DOTs, commercial vehicle enforcement agencies, Weigh-In-Motion (WIM) gate operators, traffic management centers (TMCs), and transportation research laboratories seeking automated, auditable tools for roadside screening and permit verification.

### Length of Project & Pilot Status
This repository represents an exploratory pilot research handoff developed under the U.S. Department of Transportation (USDOT) Federal Highway Administration (FHWA) DOTSI Summer Research Program.

---

## 🛠️ Prerequisites

### General Requirements
* **Operating System:** Linux (Ubuntu 20.04/22.04 recommended), Windows 10/11, or macOS.
* **Internet Connection:** Required for API-based multimodal LLM inference and downloading pre-trained weights.
* **Python Environment:** Python 3.10 or later.
* **Compute / GPU Hardware:** NVIDIA CUDA-compatible GPU (8GB+ VRAM recommended for YOLOv8 training and real-time inference).
* **Integrated Development Environment (IDE):** Databricks, VS Code, or PyCharm.

### Software & Core Dependencies
* **Computer Vision & Deep Learning Frameworks:** `torch`, `torchvision`, `ultralytics` (YOLOv8), `opencv-python`, `pillow`.
* **Multimodal Vision-Language API:** `google-generativeai` (for Gemini zero-shot prompt execution).
* **Data Processing & Geometry Libraries:** `numpy`, `pandas`, `pydantic`, `shapely`, `matplotlib`.

### Required API Credentials & Datasets
* **Google Gemini API Key:** Required for executing fine-tuned Multimodal LLM prompt scripts (`src/prompt_engine_v3.py`).
* **Commercial Vehicle Imagery Corpus:** High-resolution highway camera feeds (e.g., I-295 SB corridor dataset).
* **CVAT Annotation Exports:** CVAT XML or polyline shapefiles for training custom YOLOv8 segmentation heads.

---

## 🚀 Usage

### Repository Directory Structure
All execution scripts, pipeline tools, input/output data structures, and documentation assets are organized as follows:

```text
freight-ai-pipeline/
│
├── Code/
│   └── Freight_AI.Rproj             # RStudio / Environment project file
│
├── data/
│   ├── raw_images/                  # Raw commercial vehicle camera feeds (I-295 corridor)
│   ├── processed_dataset/           # Converted YOLO segmentation polygon masks
│   └── test_images/                 # Held-out N=50 evaluation test dataset
│
├── scripts/
│   ├── Polyline_converter.py        # Converts open CVAT polylines to closed 2D polygons
│   ├── data_filtering.py            # Preprocesses and filters raw vehicle image feeds
│   └── grid_overlay_generator.py    # Generates 1800x1000 coordinate grid overlays for VLMs
│
├── src/
│   ├── freight_compliance_validator.py  # Data quality guardrail & Shoelace math engine
│   └── prompt_engine_v3.py          # Multimodal VLM prompt execution wrapper
│
├── runs/
│   └── segment/train/weights/
│       └── best.pt                  # Trained YOLOv8-Seg PyTorch model weights
│
├── docs/                            # Schema diagrams, visual gallery, and HUD assets
├── requirements.txt                 # Python dependency manifest
└── README.md                        # Master repository documentation
