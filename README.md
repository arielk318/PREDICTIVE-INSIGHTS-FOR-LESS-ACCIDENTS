Predictive Insights for Less Accidents
Ariel Koren - 318284239
Assaf Benkormono - 318455904

Project Overview
This project aims to analyze urban traffic accident data to identify patterns and trends, predict high-risk areas, and provide actionable insights to reduce road accidents. By leveraging advanced machine learning techniques, the project offers a structured approach to accident analysis and prevention.

Features
Data Cleaning and Preprocessing: Handling missing values, normalization, encoding categorical variables, and creating custom features like risk level and injury percentages.
Predictive Modeling: Using machine learning models (Random Forest, Gradient Boosting, etc.) to predict accident-related outcomes.
Clustering Analysis: Identifying high, medium, and low-risk zones using KMeans, DBSCAN, and Agglomerative Clustering.
Visualizations: PCA and t-SNE visualizations to better understand clustering results and feature relationships.
Technologies and Libraries

Programming Language: Python

Libraries:
Data Processing: numpy, pandas, scikit-learn
Modeling: xgboost, lightgbm

Visualization: matplotlib, seaborn
Statistical Analysis: statsmodels, scipy

Dataset
The dataset contains information about urban traffic accidents, with 1,174 records and 27 features. Key features include:
SUMACCIDEN: Total accidents
DEAD: Fatalities
SEVER_INJ: Severe injuries
INJ0_19, INJ20_64, INJ65_: Injuries by age groups
PCT_DEAD, PCT_SEVER: Percentages of fatalities and severe injuries
The data was preprocessed to address missing values, normalize numerical columns, and encode categorical features.

Installation and Usage
Clone the repository:
git clone https://github.com/arielk318/PREDICTIVE-INSIGHTS-FOR-LESS-ACCIDENTS.git
Navigate to the project directory:
cd PREDICTIVE-INSIGHTS-FOR-LESS-ACCIDENTS
Install the required dependencies:
pip install -r requirements.txt
Run the Jupyter Notebook or Python scripts to explore the data and train models.
Key Files
notebooks/: Contains Jupyter Notebooks with data preprocessing, modeling, and clustering analysis.
code/: Python scripts for training models and generating results.
data/: Preprocessed and raw datasets.
report/: Final project report and documentation.
requirements.txt: Dependencies for running the project.

Results
Predictive Models: Gradient Boosting achieved the highest accuracy (82%) for predicting accident outcomes.
Clustering: Agglomerative Clustering provided the most distinct separation of risk zones with a Silhouette Score of 0.95.
Visualizations: PCA and t-SNE demonstrated clear separation of clusters and relationships between features.
Contributors

Ariel Koren: Clustering analysis, final report preparation.

Assaf Benkormono: Predictive modeling, pipeline development, and presentation preparation.

Future Directions
Explore causal relationships between features and accident severity.
Integrate external datasets to enhance prediction accuracy.
Implement deep learning techniques for improved feature extraction.

License
This project is licensed under the MIT License. See the LICENSE file for details.

