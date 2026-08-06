# Freight AI Pipeline

Commercial Freight Compliance Preprocessing Pipeline


# 🚛 Freight AI: Automated Freight Compliance & Cargo Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Seg-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-USDOT%20%2F%20FHWA-navy.svg)]()

**Automated Freight Compliance & Cargo Instance Segmentation** is an intelligent computer vision and multimodal vision-language pipeline developed for roadside freight enforcement, weigh-in-motion (WIM) gate analysis, and Over-Dimensional (OS/OW) permit compliance. 

Engineered under the **U.S. Department of Transportation (USDOT) Federal Highway Administration (FHWA) DOTSI Program**, this system combines real-time **YOLOv8-Seg** polygon extraction with zero-shot **Multimodal LLMs (Gemini)** and discrete geometric spatial surface area integration[cite: 2, 3].

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [System Architecture & Hybrid Deployment](#-system-architecture--hybrid-deployment)
- [Repository Directory Structure](#-repository-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Data Pipeline & Filtering](#-data-pipeline--filtering)
- [Prompt Engineering (V1 vs. V3)](#-prompt-engineering-v1-vs-v3)
- [Empirical Performance & Evaluation](#-empirical-performance--evaluation)
- [Usage Instructions](#-usage-instructions)
- [Institutional Continuity & Project Handoff](#-institutional-continuity--project-handoff)
- [Acknowledgements](#-acknowledgements)

---

## 🏛️ Executive Overview

Overweight commercial freight poses significant risks to highway infrastructure, accelerating pavement fatigue and bridge deterioration. State DOTs enforce strict load divisibility rules:
* **Divisible Loads:** Granular, multi-unit, or loose bulk freight (e.g., aggregate, soil, timber, gravel) subject to legal axle limits.
* **Non-Divisible Loads:** Monolithic equipment, machinery, or structural assemblies eligible for Over-Dimensional (OS/OW) state permits.

Manual inspections at physical weigh stations cause severe corridor congestion. **Freight AI** automates cargo isolation and compliance classification directly from highway camera feeds (e.g., I-295 SB corridor)[cite: 2, 3]:
1. **Pixel-Level Cargo Isolation:** Traces exact irregular payload contours while ignoring vehicle tractor cabs, tires, and surrounding passenger cars[cite: 2, 3].
2. **Spatial Surface Area Computation:** Computes 2D payload surface area from predicted polygon coordinates using Gauss's Shoelace Integration.
3. **Divisibility Classification:** Categorizes loads into `divisible`, `likely_divisible`, `non_divisible`, or `unknown` using a 5-tier rule hierarchy[cite: 2, 3].

---

## ✨ Key Features

* **YOLOv8-Seg Real-Time Edge Screening:** High-speed instance segmentation running at **12.4 ms per image** for roadside bypass gates at 55 mph.
* **Multimodal LLM Reasoning (Gemini):** High-precision zero-shot vision-language evaluation with **100% JSON schema compliance**.
* **Target Payload Isolation:** Strict annotation rules that trace exclusively cargo payload perimeters across 6 trailer body types (End-dumps, Refuse, Tankers, Flatbeds, Auto Carriers, Dry Vans).
* **Shoelace Surface Area Integration:** Discrete geometry pipeline calculating exact pixel surface area directly from predicted polygon vertices:
  $$\text{Area} = 0.5 \cdot \left\vert{} \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right\vert{}$$
* **Dynamic Data Quality Guardrails:** Automated dataset validator enforcing a strict ambiguity ceiling ($\text{Ratio}_{\text{likely\_divisible}} \le 0.15$) to prevent class imbalance collapse.

---

## ⚙️ System Architecture & Hybrid Deployment

Freight AI utilizes a **Hybrid Deployment Model** to balance low latency with deep contextual reasoning:

## 📁 Repository Directory Structure

```text
freight-ai-pipeline/
│
├── data/
│   ├── raw_images/                   # Unfiltered highway monitoring images
│   ├── processed_dataset/            # Filtered & split CVAT polygon annotations
│   └── test_images/                  # Held-out N=50 test evaluation dataset
│
├── scripts/
│   ├── Polyline_converter.py         # Converts open CVAT polylines to closed YOLO polygons
│   ├── data_filtering.py             # Data filtering & quality audit script
│   └── grid_overlay_generator.py     # Generates high-density coordinate grids for VLM prompts
│
├── src/
│   ├── freight_compliance_validator.py # Class-balance validator & Shoelace math engine
│   └── prompt_engine_v3.py           # Fine-tuned Multimodal LLM prompt wrapper
│
├── runs/
│   └── segment/train/weights/
│       └── best.pt                   # Fine-tuned YOLOv8-Seg PyTorch weights
│
├── docs/                             # Documentation assets and schema diagrams
├── requirements.txt                  # Python dependencies
└── README.md                         # Repository documentation

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
