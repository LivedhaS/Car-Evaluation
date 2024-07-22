# Car-Evaluation
The objective of this project is to build a deep learning model that predicts the evaluation 
of cars (unacceptable, acceptable, good, very good) based on various features such as 
buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, 
and safety rating. The Car Evaluation dataset from the UCI Machine Learning Repository 
is used for this purpose. 

# Dataset Overview
The Car Evaluation dataset contains the following attributes: 
1. buying: Buying price (v-high, high, med, low)
2. maint: Maintenance price (v-high, high, med, low)
3. doors: Number of doors (2, 3, 4, 5-more)
4.  persons: Passenger capacity (2, 4, more)
5.  lug_boot: Luggage boot size (small, med, big)
6.  safety: Safety rating (low, med, high)
7.  class: Car evaluation (unacc, acc, good, vgood).
8.   The dataset consists of 1728 instances, each with the above attributes.

# Data Preprocessing
1. Loading the Dataset: The dataset was loaded into a Pandas DataFrame from a 
CSV file.
2. Handling Missing Values: The dataset did not contain any missing values.
3. Encoding Categorical Variables: All categorical features were encoded to 
numeric values using `LabelEncoder`.
4. Splitting the Data: The data was split into training and test sets using an 80-20 
split.
5. Standardizing the Features: The features were standardized using 
`StandardScaler` to have a mean of 0 and a standard deviation of 1.

# Model Architecture
The neural network model was built using TensorFlow and Keras. The architecture consisted of: 
1. Input Layer: Matching the number of features in the dataset. 
2. Hidden Layers: Two hidden layers with 64 and 32 neurons respectively, both 
using ReLU activation function. 
3. Output Layer: Four neurons with a softmax activation function to predict the 
probability of each class. 
 
#  Model Compilation
The model was compiled using the Adam optimizer, sparse 
categorical cross-entropy loss function, and accuracy as the evaluation metric. 
 
#  Model Training
The model was trained using the training set for 50 epochs with a 
batch size of 32. 20% of the training data was used for validation during training. 
 
# Model Evaluation 
The model was evaluated using the test set, achieving an 
accuracy of approximately 95%. 
 
# Training History
The training history showed that both the training and validation 
accuracy increased over epochs, while the training and validation loss decreased, 
indicating that the model was learning effectively without overfitting. 
