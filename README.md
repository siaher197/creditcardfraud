# creditcardfraud
objective is to create the best classifier for credit card fraud detection. To do it, we'll compare classification models from different methods :
Logistic regression
Support Vector Machine
Bagging (Random Forest)
Boosting (XGBoost)
Neural Network (tensorflow/keras)
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. I decided to proceed to an undersampling strategy to re-balance the class.
Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.


We need to standardize the 'Amount' feature before modelling. For that, we use the StandardScaler function from sklearn.
(https://user-images.githubusercontent.com/109465506/185756968-112c1940-a9ac-43da-9d84-c76c6195d157.png)




        
