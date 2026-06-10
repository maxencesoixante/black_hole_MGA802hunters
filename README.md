# Automated Astronomical Anomaly Detector 

**Session Project - MGA 802: Introduction to Programming with Python**

## Project Description
This project is an object-oriented Python library designed to automate the processing, analysis, and classification of stellar light curves. 
<img width="999" height="691" alt="image" src="https://github.com/user-attachments/assets/5ad33a7c-ec4a-4ebd-8039-eb3aad4ee2f7" />


Inspired by *Citizen Science* methodologies, our algorithm aims to detect astronomical anomalies (exoplanets, gravitational microlensing from black holes, and stellar flares) within raw data obtained from space telescopes (KEPLER & TESS).
<img width="6016" height="4016" alt="TESS_alone_high_res (1)" src="https://github.com/user-attachments/assets/7a09a459-5070-4a32-b8d5-50f49f39b129" />


## Main Features
- **Automated Extraction:** Connection and download of real data via the NASA API (`lightkurve`).
- **Preprocessing:** Time-series cleaning (smoothing, handling missing data/observation gaps).
- **Algorithmic Detection:** Identification of anomalous brightness variations against a baseline.
- **Classification:** Shape analysis of the anomaly (Sudden peak, U-shaped dip, Symmetric bell) to deduce the event type.
- **Visualization:** Generation of clear plots highlighting the detections for the user.

## Team and Task Allocation

To ensure equitable participation and cover all course requirements (data manipulation, algorithms, software architecture), tasks are distributed as follows:

* **Maxence: Data Engineering and Preprocessing**
  * Implementation of the API connection module (fetching `.fits` files or Kaggle data).
  * Development of mathematical functions for smoothing and normalizing the light flux.
  * Creation of logic to ignore data "gaps" (telescope observation interruptions).

* **Jules: Algorithm and Modeling (Machine Learning / Statistics)**
  * Definition of alert thresholds for initial anomaly detection.
  * Implementation of the classification engine to differentiate the 3 events (Flare, Exoplanet Transit, Black Hole).
  * Calculation of the detection probability or confidence index.

* **Alexandre: Architecture, User Interface (CLI), and Visualization**
  * Overall code structuring using Object-Oriented Programming (creating the distributable module).
  * Development of user interaction (handling input parameters like target ID, dates, etc.).
  * Creation of the plotting module (Matplotlib/Seaborn) to display before/after curves and highlight anomalies.

## Installation and Usage
*(To be completed during the development phase)*
1. Clone the repository: `git clone [Repo URL]`
2. Install dependencies: `pip install -r requirements.txt`
3. Run a sample analysis: `python main.py --target KIC_XXXXXXX`

## References and Dependencies
- [[Lightkurve Documentation](https://docs.lightkurve.org/)](https://lightkurve.github.io/lightkurve/reference/search.html)
- Reference about stellar flare morphology with TESS across the main sequence : https://www.aanda.org/articles/aa/full_html/2025/02/aa52489-24/aa52489-24.html
- Numpy, Pandas, Matplotlib, Scikit-Learn
