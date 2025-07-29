# Electric Motor Temperature Prediction

## Project Overview

This project focuses on **predicting the temperature of an electric rotor motor** using various operational parameters and sensor data. The primary goal is to develop a robust machine learning model capable of accurately estimating the rotor temperature. This predictive capability is crucial for:
* **Optimizing motor performance:** Ensuring the motor operates within safe and efficient temperature ranges.
* **Preventing overheating:** Proactively identifying potential overheating issues to avoid damage and downtime.
* **Extending motor lifespan:** Implementing intelligent control strategies based on temperature predictions.
* **Enabling proactive maintenance:** Shifting from reactive to predictive maintenance strategies.

## Dataset

The dataset used in this project is the "Electric Motor Temperature" dataset, publicly available on Kaggle. It contains sensor readings from a permanent magnet synchronous motor (PMSM) deployed on a test bench, comprising multiple measurement sessions.

* **Dataset Name:** Electric Motor Temperature
* **Source:** [Kaggle](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)
* **File Used:** `measures_v2.csv` (or `pmsm_temperature_data.csv` if that's the full name of the file within the Kaggle download)

### Dataset Description (from Kaggle)

The dataset comprises several sensor data collected from a permanent magnet synchronous motor (PMSM) deployed on a test bench. All recordings are sampled at 2 Hz. The data set consists of multiple measurement sessions, which can be distinguished from each other by the "profile_id" column. A measurement session can be between one and six hours long. The motor is excited by hand-designed driving cycles denoting a reference motor speed and a reference torque. Currents in d/q-coordinates (columns "i_d" and "i_q") and voltages in d/q-coordinates (columns "u_d" and "u_q") are a result of a standard control strategy trying to follow the reference speed and torque. Columns "motor_speed" and "torque" are the resulting quantities achieved by that strategy, derived from set currents and voltages.

**Key Features (among others):**
* `ambient`: Ambient temperature (°C)
* `coolant`: Coolant temperature (°C)
* `u_d`, `u_q`: Voltage d/q-components (V)
* `i_d`, `i_q`: Current d/q-components (A)
* `motor_speed`: Motor speed (rpm)
* `torque`: Torque (Nm)
* `profile_id`: Identifier for measurement session
* `pm`: Permanent magnet temperature (Rotor temperature) (°C) - **This is a primary target variable.**
* `stator_yoke`, `stator_tooth`, `stator_winding`: Stator temperatures (°C)
