# Spotify-Popularity-Prediction
Predicting Spotify song popularity using Machine Learning (R)


# 🎵 Spotify Song Popularity Prediction using Machine Learning (R)

## 📘 Project Overview  
This project aims to predict the **popularity of songs on Spotify** using various **audio features** such as danceability, energy, loudness, tempo, and more.  
The model helps in understanding what kind of musical characteristics contribute to a song becoming popular.

I built this project using **R language** and **Machine Learning techniques**, focusing on model accuracy, interpretability, and visualization.

---

## 🧠 Objective  
- To analyze Spotify dataset and find patterns in audio features.  
- To build a predictive model for song popularity.  
- To visualize feature importance and relationships between variables.

---

## 🧰 Tech Stack  
- **Programming Language:** R  
- **Libraries Used:**  
  - `tidyverse`  
  - `ggplot2`  
  - `caret`  
  - `randomForest`  
  - `corrplot`  

---

## 📂 Dataset  
The dataset contains information about Spotify tracks with features like:  
- `danceability`  
- `energy`  
- `loudness`  
- `tempo`  
- `valence`  
- `popularity` (Target variable)  

> **Note:** The CSV file (`spotify_data.csv`) is already included inside the project folder.  
> You just need to ensure the file name and path match while running the R script.

---

## ⚙️ Project Workflow  
1. **Data Preprocessing:**  
   - Handle missing values  
   - Remove duplicates  
   - Normalize numerical features  

2. **Exploratory Data Analysis (EDA):**  
   - Visualized correlations and feature distributions  
   - Identified key variables affecting popularity  

3. **Model Building:**  
   - Used **Random Forest** and **Linear Regression** models  
   - Compared model accuracy and RMSE  

4. **Model Evaluation:**  
   - Evaluated performance using RMSE, MAE, and R² metrics  
   - Selected best-performing model  

---

## 📊 Results  
- Achieved high predictive accuracy using **Random Forest**.  
- Identified top features impacting song popularity.  
- Created visualizations for better interpretability.  

---

## 🚀 Future Scope  
- Incorporate **Deep Learning models** for advanced predictions.  
- Integrate **Spotify API** for real-time data updates.  
- Build a simple **web dashboard** for user interaction.

---

⭐ If you like this project, don’t forget to **star the repo** and follow for more Data Science projects!
