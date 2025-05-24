# Infant Drowning Detection Using Video Vision Transformer (ViViT)

## Abstract
Drowning is the third leading cause of accidental fatalities worldwide, with indoor swimming pool drownings, especially among children, becoming a significant concern. Traditional monitoring methods are often insufficient for large pools, leading to increased drowning incidents.

This project proposes an innovative drowning detection system using Video Vision Transformer (ViViT) to monitor swimming pools. The system extracts spatio-temporal features from video clips using a novel embedding scheme and transformer variants to accurately identify drowning events in real-time.

Leveraging a pre-trained ViViT model and cosine similarity measures, the system efficiently detects infant drowning behaviors without relying on heavy resource-intensive models. It utilizes Temporal Transformer and Feature Pyramid Networks to maintain high accuracy while ensuring real-time performance.

The ultimate goal is to provide automated alerts for caregivers to reduce drowning accidents and improve overall water safety.

## Objectives
- Develop a ViViT-based model to detect infant behavior around water bodies.
- Establish real-time camera monitoring near pools or other water sources.
- Implement ViViT to recognize safe vs. dangerous infant behaviors.
- Generate immediate alerts for caregivers upon detecting drowning risk.
- Design an intuitive interface for caregivers to monitor infants effectively.
- Validate system performance and accuracy in real-world scenarios.

## Features
- Real-time video analysis using ViViT transformer model.
- Lightweight architecture optimized for quick detection.
- Automated alert system for immediate caregiver notification.
- User-friendly monitoring interface.
- Support for infant-specific behavior recognition near water.

## Technologies Used
- Python
- Video Vision Transformer (ViViT)
- Temporal Transformer & Feature Pyramid Networks
- OpenCV (for video processing)
- Alert system integration (e.g., OLED display, notifications)
- Web interface (optional for caregiver monitoring)

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/sanjevi06/infant-drowning-detection.git
