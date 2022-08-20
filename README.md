Credit Card Fraud Detection
objective is to create the best classifier for credit card fraud detection. To do it, we'll compare classification models from different methods : Logistic regression Support Vector Machine Bagging (Random Forest) Boosting (XGBoost) Neural Network (tensorflow/keras) The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. I decided to proceed to an undersampling strategy to re-balance the class. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
We need to standardize the 'Amount' feature before modelling. For that, we use the StandardScaler function from sklearn. 
 
The dataset is highly imbalanced ! It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, I will proceed to random undersampling.
The simplest undersampling technique involves randomly selecting examples from the majority class and deleting them from the training dataset. This is referred to as random undersampling.
Although simple and effective, a limitation of this technique is that examples are removed without any concern for how useful or important they might be in determining the decision boundary between the classes. This means it is possible, or even likely, that useful information will be deleted.
 
Our dataset is now perfectly balanced !
The last step before modelling is now to split the data intro train and test samples. The test set will be composed of 20% of the data.
We will use the train dataset to train our models and then evaluate them of the test set :

 1.Logistic Regression
Accuracy Logit: 0.9459459459459459
Precision Logit: 0.9509803921568627
Recall Logit: 0.8981481481481481
F1 Score Logit: 0.9238095238095237
![card logistic matrix](https://user-images.githubusercontent.com/109465506/185758386-b3afacbe-1540-402e-b9d7-f59cb4d39ae2.png)
AUC Logistic Regression : 0.9742907801418439
![card roc](https://user-images.githubusercontent.com/109465506/185758422-df8080bf-cd98-435d-82dd-4cdd9ba65210.png)
![card precision recall curve](https://user-images.githubusercontent.com/109465506/185758436-12ff9bd5-3212-47a7-9a79-f989e9f67a0d.png)
 Classification metrics for Logistic Regression (rounded down) :
•	Accuracy : 0.94
•	F1 score : 0.92
•	AUC : 0.96

2. Support Vector Machine

Accuracy SVM: 0.9425675675675675
Precision SVM: 0.9789473684210527
Recall SVM: 0.8611111111111112
F1 Score SVM: 0.9162561576354681
![card svm matrix](https://user-images.githubusercontent.com/109465506/185758470-eadb0188-f47a-4951-91f9-7a4bc0d99c48.png)
AUC SVM : 0.9758175728920409
![card svm roc](https://user-images.githubusercontent.com/109465506/185758480-8203f0a5-5117-453b-b738-b558b6254ff5.png)
![crad svm roc recall curve](https://user-images.githubusercontent.com/109465506/185758499-86f9d517-ac4a-49dd-ae31-953820119e18.png)
Classification metrics for SVM (rounded down) :
•	Accuracy : 0.94
•	F1 score : 0.92
•	AUC : 0.97

3. Ensemble learning : Bagging (Random Forest)
Train the model, predict, score
Accuracy RF: 0.9425675675675675
Precision RF: 0.9690721649484536
Recall RF: 0.8703703703703703
F1 Score RF: 0.9170731707317072
![card random matrix](https://user-images.githubusercontent.com/109465506/185758537-3e0d82be-cf8c-433d-866c-d91e3f08afbe.png)
AUC Random Forest : 0.969144011032309
![card random roc curve](https://user-images.githubusercontent.com/109465506/185758566-3d5cdc42-b028-4445-9027-9379be1f97b7.png)
![card random recall curve](https://user-images.githubusercontent.com/109465506/185758575-e19436f1-5693-42d2-95a6-a1b9f14d7f9e.png)
Classification metrics for Random Forest (rounded down) :
•	Accuracy : 0.95
•	F1 score : 0.93
•	AUC : 0.97



4. Ensemble learning : Boosting (XGBoost)
The sequential ensemble methods, also known as “boosting”, creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence. The first model is built on training data, the second model improves the first model, the third model improves the second, and so on.

Accuracy XGB: 0.9459459459459459
Precision XGB: 0.96
Recall XGB: 0.8888888888888888
F1 Score XGB: 0.923076923076923
![crad XG boost matrix](https://user-images.githubusercontent.com/109465506/185758587-a93bb5cf-73e3-4d2c-b9cf-500bf6cfe20d.png)
AUC XGBoost : 0.9766055949566588
 ![Card XGboost roc curve](https://user-images.githubusercontent.com/109465506/185758661-7b0396db-465f-4e85-b650-c54afda66b0a.png)
![Card XGboost roc recall](https://user-images.githubusercontent.com/109465506/185758680-3e1f98a3-3873-4da6-8163-1b3fd0961f87.png)
Classification metrics for XGBoost (rounded down) :
•	Accuracy : 0.95
•	F1 score : 0.93
•	AUC : 0.97

5. Multi Layer Perceptron

Accuracy MLP: 0.9493243243243243
Precision MLP: 0.9345794392523364
Recall MLP: 0.9259259259259259
F1 Score MLP: 0.9302325581395349

![card matrix mlp](https://user-images.githubusercontent.com/109465506/185758720-e3c68adf-aa4e-4572-aba0-1e2982d4c8f5.png)
AUC MLP : 0.9843380614657209
![crad mlp roc curve](https://user-images.githubusercontent.com/109465506/185758757-d8d21cb2-9a48-4c1e-97e2-aa5e6a50b17b.png)
![card mlp recall curve](https://user-images.githubusercontent.com/109465506/185758772-65d39e83-f620-4d69-93f6-3aee7aaf85d8.png)
Classification metrics for Multi Layer Perceptron (rounded down) :
•	Accuracy : 0.95
•	F1 score : 0.94
•	AUC : 0.98


6. Multilayer Neural Network with Tensorflow/Keras

Accuracy Neural Net: 0.9358108108108109
Precision Neural Net: 0.9320388349514563
Recall Neural Net: 0.8888888888888888
F1 Score Neural Net: 0.909952606635071

![card nn matrix](https://user-images.githubusercontent.com/109465506/185758796-84430e85-e06e-4422-b99e-fe0b80fc470e.png)
 AUC Neural Net:  0.9721729708431834
![card nn roc curve](https://user-images.githubusercontent.com/109465506/185758806-86703bd2-0609-45a8-af69-a337e7e4aac6.png)
![card nn recall curve](https://user-images.githubusercontent.com/109465506/185758811-99c1f936-e97e-42a4-b431-4f7337191efc.png)

 Classification metrics for Neural Network (rounded down) :
•	Accuracy : 0.95
•	F1 score : 0.94
•	AUC : 0.98


Conclusion:
The multilayer neural network has the best performance according to our three most important classification metrics (Accuracy, F1-score and AUC). The Multi Layer Perceptron from sklearn is the one that minimizes the most the false negatives so I decided to keep this model to predict credit card frauds. It's very important that a bank do not miss frauds so minimizing false negatives rate is essential.

        
