# Smart Chilli Leaf Disease Detection System



> A solar-powered IoT device using ESP32-CAM and TinyML that detects 

> 8 chilli leaf diseases and sends SMS alerts to farmers in Andhra Pradesh 

> — costing under ₹2,500 with zero internet required.





##### **What This Project Does**



Small chilli farmers in AP lose 30–70% yield every season to diseases 

because they have no affordable automated detection system. This device 

continuously monitors chilli leaves, identifies disease using AI, and 

sends the disease name + exact pesticide recommendation directly to the 

farmer's basic mobile phone via SMS.





##### **5 Novel Features**



**Feature  and Description** 



1\.Predictive Early Warning : Warns farmer 6–12 hrs BEFORE disease crosses threshold 

2\.Crop Stage Adaptive Thresholds : Auto-adjusts sensitivity based on growth stage 

3\.Enclosed LED Illumination Chamber : Consistent lighting for reliable colour analysis 

4\.Disease Recovery Verification : Confirms if treatment worked within 18 hours 

5\.Self-Learning Threshold (EMA) : Calibrates to each field over time 





##### **ML Model**



**Property and  Value** 



Architecture :  MobileNetV2 (alpha=0.35) 

Training method : Transfer Learning (ImageNet weights) 

Dataset : 4,800 images, 8 classes 

Validation accuracy : 87.4%

TFLite model size : 766 KB (INT8 quantized) 

Inference device : ESP32-CAM (240MHz, 520KB RAM) 

Training platform : Google Colab (free GPU) 





##### **8 Diseases Detected**



1\. Anthracnose

2\. Bacterial Spot  

3\. Cercospora Leaf Spot

4\. Healthy

5\. Leaf Curl Virus

6\. Mosaic Virus

7\. Powdery Mildew

8\. Thrips Damage (Black Thrips - Scirtothrips dorsalis)





##### **Hardware (Total: ₹2,500)**



| Component | Purpose | Cost |



| ESP32-CAM AI-Thinker | Main controller + camera | ₹380 |

| SIM800L GSM Module | SMS alerts via 2G | ₹350 |

| 18650 Battery + Holder | Power SIM800L | ₹180 |

| Solar Panel 6V 1W | Field power | ₹180 |

| Breadboard + Wires | Prototype | ₹110 |

| LEDs + Resistors + Button | Indicators | ₹25 |

| Jio/Airtel SIM (2G) | GSM network | ₹10 |



##### **How it Works**



**Mirchi Leaf**

**↓**

**ESP32-CAM captures 96×96 image inside LED chamber**

**↓**

**Algorithm 1: Blob Detection → counts dark lesion spots**

**Algorithm 2: HSV Colour Analysis → measures damage %**

**Algorithm 3: TinyML MobileNetV2 → classifies disease**

**↓**

**Final decision combining all 3 signals**

**↓**

**SIM800L sends SMS to farmer's basic mobile phone**

**"Disease: Thrips Damage | Damage: 18% | Spray Spinosad 45SC 1ml/L"**



**---**

**Training Graph**



**!\[Training Accuracy Graph](images/training\_graph.png)**





**Technologies Used**



**- TensorFlow Lite Micro (TinyML on ESP32)**

**- MobileNetV2 Transfer Learning**

**- HSV Colour Analysis (OpenCV approach)**

**- Blob Detection (Connected Component Analysis)**

**- Embedded C++ / Arduino IDE**

**- Google Colab GPU Training**

**- SIM800L AT Commands**

**- ESP32 Deep Sleep (10 microamps between captures)**





**Project Type**



**EPICS — Engineering Projects in Community Service**  

**Department: Computer Science Engineering**  

**Target: Small and marginal chilli farmers, Andhra Pradesh, India**  

**Districts: Guntur, Prakasam, Krishna, Khammam**





**Dataset**



**See `dataset\_info/dataset\_description.txt` for dataset details.**  

**Original source: \[PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)**



**---**



**How to Run**

**Training (Google Colab):**

**1. Open `colab\_training/COLAB\_training\_complete.py`**

**2. Run on Google Colab with GPU enabled**

**3. Downloads `chilli\_disease.tflite` and `chilli\_model.h`**



**Arduino Deployment:**

**1. Open `arduino\_code/chilli\_detector\_FINAL.ino` in Arduino IDE**

**2. Place `chilli\_disease.tflite` in `arduino\_code/data/` folder**

**3. Tools → Board → AI Thinker ESP32-CAM**

**4. Tools → ESP32 Sketch Data Upload (uploads model to ESP32)**

**5. Click Upload**

**6. Open Serial Monitor at 115200 baud**



**File Structure:**

**Chilli\_Disease\_Detection/**

**├── README.md**

**├── arduino\_code/**

**│   └── chilli\_detector\_FINAL.ino**

**├── colab\_training/**

**│   └── COLAB\_training\_complete.py**

**├── model/**

**│   ├── chilli\_disease.tflite**

**│   ├── chilli\_model.h**

**│   └── training\_graph.png**

**├── images/**

**│   ├── training\_graph.png**

**│   └── hardware\_setup.jpg**

**└── dataset\_info/**

**└── dataset\_description.txt**

