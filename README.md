# Deep-Learning-Challenge (Module_21)

## Overview of the Analysis
The nonprofit foundation Alphabet Soup wants a tool to help select applicants for funding with the best chance of success. By using machine learning and neural networks, we aim to create a binary classifier that predicts the success of funding applicants. Using a dataset of over 34,000 organizations funded by Alphabet Soup, we analyzed various features and built a deep learning model to predict the success of charitable donations. The primary goal is to optimize this model to achieve an accuracy higher than 75%.

## Results

### Data Preprocessing
- **Target Variable(s):**
  - The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the charity donation was successful.
  
- **Feature Variable(s):**
  - The feature variables include `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.

- **Variables to Remove:**
  - The `EIN` and `NAME` columns were removed from the input data as they are identifiers and do not provide predictive value for the model.

### Compiling, Training, and Evaluating the Model
- **Neurons, Layers, and Activation Functions:**
  - **First Hidden Layer:** 80 neurons, ReLU activation function
  - **Second Hidden Layer:** 30 neurons, ReLU activation function
  - **Output Layer:** 1 neuron, Sigmoid activation function

  These choices were made based on the need to capture complex patterns in the data while preventing overfitting. The ReLU activation function is commonly used for hidden layers due to its ability to handle non-linearity, while the Sigmoid function is suitable for binary classification problems.

- **Model Performance:**
  - The initial model did not achieve the target performance of 75% accuracy. So, the model needs optimization to get better perfomance (>75%)

- **Optimization Steps:**
  - **Adjusting Input Data:**
    - Binning rare occurrences in categorical variables.
    - Dropping non-informative features `SPECIAL_CONSIDERATIONS`.
  
  - **Model Architecture:**
    - Increasing the number of neurons in the hidden layers.
    - Adding an additional hidden layer.
    - Use different activation function for the third attempt (SGD)
  
  - **Training Regimen:**
    - Increasing the number of epochs.
    - Implementing callbacks for early stopping and saving model weights.

### Summary
- The deep learning model built for Alphabet Soup Charity showed improvement through various optimization techniques, though achieving an accuracy higher than 75% was challenging. Three optimization attempts were made to improve the model's accuracy. Details are here below;
  - **Before Optimization:** Evaluation report is Loss = 0.5617 and Accuracy = 0.72968 ( <img width="745" alt="AlphabetSoupCharity" src="https://github.com/user-attachments/assets/5f87ccd5-45e1-4327-bc9e-c200c8477527">)
  - **Optimization attemt 1:** Evaluation report is Loss = 0.570778 and Accuracy = 0.72816 ( <img width="696" alt="AlphabetSoupCharity_Optization_1" src="https://github.com/user-attachments/assets/2e1b02b2-8dd7-47cf-b0b5-b38610021beb">)
  - **Optimization attemt 2:** Evaluation report is Loss = 0.5789 and Accuracy = 0.7278 ( <img width="767" alt="AlphabetSoupCharity_Optization_2" src="https://github.com/user-attachments/assets/f64c2a4e-f15a-4a48-86a2-0078ff392d38">)
  - **Optimization attemt 3:** Evaluation report is Loss = 0.5554 and Accuracy = 0.73148 ( <<img width="721" alt="AlphabetSoupCharity_Optization_3" src="https://github.com/user-attachments/assets/38de7d8b-e1bc-4191-a416-fbcb0388c36c">)

**Recommendation:**
To further improve the model's performance, I recommend experimenting hyperparameter tuning can capture more complex relationships in the data and potentially achieve the target accuracy. 
