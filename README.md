# IEEE-CIS Fraud Detection

##  კონკურსის მიმოხილვა
კონკურსის მიზანია ონლაინ ტრანზაქციების თაღლითობის აღმოჩენა. მოცემულია ორი dataset — transaction და identity, რომლებიც TransactionID-ით არის დაკავშირებული. სამიზნე ცვლადია isFraud (0 ან 1), სადაც მონაცემები დაუბალანსებელია.

## რეპოზიტორიის სტრუქტურა
Cleaning, Feature Engineering, Feature Selection, Training, გატესტვა თითოეულ მოდელზე
```
ML_Assignment_2/
├── model_experiment_LogisticRegression.ipynb ← Logistic Regression ექსპერიმენტები
├── model_experiment_DecisionTree.ipynb ← Decision Tree ექსპერიმენტები
├── model_experiment_RandomForest.ipynb ← Random Forest ექსპერიმენტები
├── model_experiment_XGBoost.ipynb ← XGBoost ექსპერიმენტები
├── model_inference.ipynb ← საუკეთესო მოდელის ატვირთვა Registry-ში. submission.
└── README.md ← რეპოზიტორიის აღწერა
```
## Feature Engineering

### NaN მნიშვნელობების დამუშავება
რიცხვითი ცვლადებისთვის გამოვიყენეთ **median** შევსება. IEEE Fraud Detection-ის რიცხვითი feature-ბი (TransactionAmt, card1-6) ძალიან არის გადახრილი და ასიმეტრიულად განაწილებული. შესაბამისად მედიანით შევსება შუა მნიშვნელობებს მოგვცემდა და უფრო ზუსტი იქნებოდა, ვიდრე საშუალო ან მოდა. Decision Tree და Random Forest-ისთვის დავამატეთ **Outlier Removal** IQR მეთოდით TransactionAmt ცვლადზე.


### კატეგორიული ცვლადების რიცხვითში გადაყვანა
ყველა notebook-ში გამოვიყენეთ LabelEncoder კატეგორიული ცვლადების დასაშიფრად.

## გამოვიყენეთ 4 განსხვავებული მოდელი

### Cleaning

| მოდელი | Null Threshold | დამატებითი Cleaning |
|---|---|---|
| Logistic Regression | 0.6 | — |
| Decision Tree | 0.5 | Outlier Removal (IQR 3x) |
| Random Forest | 0.5 | Outlier Removal (IQR 3x) |
| XGBoost | 0.7 | — |

XGBoost-ისთვის threshold უფრო მაღალია (0.7), რადგან XGBoost NaN მნიშვნელობებს თავად ართმევს და მეტი feature-ის შენარჩუნება სასარგებლოა.

## Feature Selection

დავამატეთ 3 ახალი feature.

**SelectKBest** — სტატისტიკური ტესტით (f_classif) ირჩევს k საუკეთესო feature-ს. სწრაფი და ეფექტური მეთოდია linear მოდელებისთვის.

**Feature Importances** — Decision Tree-ს საკუთარი importance scores-ის გამოყენება. tree-based მოდელებისთვის ბუნებრივი მიდგომაა.

**Correlation Threshold** — target ცვლადთან კორელაციის მიხედვით feature-ების გაფილტვრა. მარტივი მაგრამ ეფექტური მეთოდია XGBoost-ისთვის.

## Training

| მოდელი | საუკეთესო AUC | Imbalanced გადაწყვეტა |
|---|---|---|
| Logistic Regression | 0.7289 | class_weight=balanced |
| Decision Tree | 0.8595 | class_weight=balanced |
| Random Forest | 0.8830 | class_weight=balanced |
| XGBoost | 0.9623 | scale_pos_weight |

### Hyperparameter ოპტიმიზაციის მიდგომა

თითოეული მოდელისთვის გავტესტეთ რამდენიმე hyperparameter კომბინაცია:

**Logistic Regression:**
- C პარამეტრი (regularization): 1.0, 10.0
- solver: lbfgs, saga
- penalty: l2, l1
- საუკეთესო: C=1.0, class_weight=balanced (AUC: 0.7289)
  
ამ მოდელმა მოგვცა ყველაზე დაბალი შედეგი. ეს underfitting-ის მაგალითია. linear მოდელი ვერ იჭერს fraud-ის რთულ არაწრფივ pattern-ებს.

**Decision Tree:**
- max_depth: None (baseline), 5, 10
- criterion: gini, entropy
- min_samples_leaf: default, 50
- საუკეთესო: max_depth=10, entropy, class_weight=balanced (AUC: 0.8595)
  
max_depth=None-ს (baseline) მაღალი train AUC ჰქონდა მაგრამ validation-ზე დაბალი — overfitting-ის მაგალითი.

**Random Forest:**
- n_estimators: 100, 200
- max_depth: None, 10, 15
- min_samples_leaf: default, 20
- საუკეთესო: n_estimators=200, max_depth=15, class_weight=balanced (AUC: 0.8830)
  
Decision Tree-ზე უკეთესი შედეგი, რადგან ბევრი ხის საშუალება variance-ს ამცირებს. n_estimators=200 უფრო სტაბილური შედეგი იძლევა ვიდრე 100.

**XGBoost:**
- n_estimators: 100, 300, 500
- learning_rate: 0.3, 0.01, 0.05
- subsample და colsample_bytree: 0.8
- საუკეთესო: n_estimators=500, learning_rate=0.05, subsample=0.8 (AUC: 0.9623)
  
scale_pos_weight-მა imbalanced data-ს პრობლემა ეფექტურად გადაჭრა. subsample და colsample_bytree=0.8-მა overfitting შეამცირა

## საბოლოო მოდელად შეირჩა **XGBoost** 

მიზეზი იყო ყველაზე მაღალი validation AUC, overfitting-ის კონტროლი და imbalanced data-ს კარგი დამუშავება.

## MLflow ექსპერიმენტები DagsHub-ზე

ყველა run რეგისტრირებულია საიტზე:
https://dagshub.com/skoba23/ML_Assignment_2.mlflow/#/experiments
