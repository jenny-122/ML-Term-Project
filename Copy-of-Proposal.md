# Machine Learning Project Proposal
1.	Project Type: Research-flavor project
2.	Problem addressed: This research has the potential to inform decision-making processes at various levels of the healthcare system, leading to improved resource allocation, policy development, and patient care outcomes.
3.	Project Goal and Motivation: The study aims to forecast future medication expenditure using data from the Medicaid by Drug dataset. This research seeks to provide insights into the expected costs of common medications for eligible patients requiring financial assistance.
4.	Methodology:
    a.	Data Preparation: Ensure the dataset is clean, organized, and ready for analysis. 
    b.	Feature Selection: Identify relevant features that could impact medication costs, such as drug characteristics, manufacturer information, and historical spending trends.
    c.	Model Selection: Select suitable machine learning algorithms for predicting medication costs. 
    d.	Training and Testing: Split the dataset into training and testing sets to train the model on historical data and evaluate its performance on new data.
    e.	Model Training: Train the chosen machine learning model using features like average spending per dosage unit and changes in spending over time to predict future medication costs.
    f.	Model Evaluation: Assess the model's performance using metrics.
    g.	Prediction: Use model’s evaluation to make predictions on future medication costs based on new data for upcoming years.
5.	Datasets used: 
    a.  https://healthdata.gov/dataset/Medicaid-Spending-by-Drug/bzpg-gf5q/about_data
    b.	https://healthdata.gov/dataset/Drug-Products-in-the-Medicaid-Drug-Rebate-Program/rcra-yvyi/about_data
6.	Resources needed to carry out your project:
    a.	NumPy: For numerical computations and data manipulation.
    b.	Pandas: For data manipulation, cleaning, and analysis.
    c.	Scikit-learn (sklearn): For implementing machine learning models and performance evaluation.
    d.	PyTorch: If deep learning models are considered for prediction tasks.
7.	Group Members: Jenny Dinh (solo)

## Milestones:
- Week 1:
    1.	Data Collection: Gather the Medicaid by Drug dataset or similar data sources.
    2.	Data Preparation: Clean the dataset, handle missing values, encode categorical variables, and scale numerical features.
- Week 2:
    1.	Feature Selection: Identify relevant features that could influence medication costs.
    2.	 Exploratory Data Analysis (EDA): Conduct preliminary analysis to understand the dataset and visualize trends.
- Week 3:
    1.	Model Selection: Choose suitable machine learning algorithms for predicting medication costs.
    2.	Train and Test: Split the dataset into training and testing sets, and train initial models using basic regression algorithms.
- Week 4:
    1.	Model Evaluation: Evaluate model performance using metrics such as Mean Absolute Error, Mean Squared Error, or R-squared.
    2.	Refinement: Fine-tune model hyperparameters and feature selection based on initial evaluation results.	
- Week 5:
    1.	Final Model Training: Train the final machine learning model using optimized features and parameters.
    2.	Prediction: Use the trained model to make predictions on future medication costs based on new data.
    3.	Model Evaluation: Evaluate the final model's performance on test data and compare it with initial results.
    4.	Documentation: Document the research process, including data preparation steps, model selection, and evaluation metrics.
