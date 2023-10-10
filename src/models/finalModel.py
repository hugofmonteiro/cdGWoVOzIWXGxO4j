import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

### Uniquely and for the sole purpose of learning ML imputation for missing data, 
#### this job was performed using these newly imputated values. Keep in mind that 
#### f1-score would be the same if the variables that contain missing data 
#### (job, education, and contact) were dropped for the random under-sampling technique. 
#### Both ways were tested and this is the longer final model with ML imputation.

# impoting df
df = pd.read_csv('../data/term-deposit-marketing-2020.csv')

# split the df into label and features
label = df.loc[:, ['y']]
features = df.drop('y', axis=1)

# apply OHE to label
for col in columns_to_encode:
    encoder = LabelEncoder()
    label['y'] = encoder.fit_transform(label['y'])
    
# perform imputation of missing values for job, education, and contact through row machine learning prediction
# first create a copy of the dataframe
df_predict = features.copy()

# create new_df_features and the separate target DataFrames
new_df_features = df_predict.drop(columns=['job', 'education', 'contact'])
job_df = df_predict[['job']]
education_df = df_predict[['education']]
contact_df = df_predict[['contact']]

def encode_features(features_df):
    encoders = {}  # Store encoders for potential reverse encoding
    for column in features_df.columns:
        # Check if the column is of object type (indicative of a categorical variable)
        if features_df[column].dtype == 'object':
            le = LabelEncoder()
            non_nan_values = features_df[column].dropna()
            le.fit(non_nan_values)
            encoders[column] = le  # Store the trained encoder
            
            # Only transform the non-NaN values
            features_df.loc[non_nan_values.index, column] = le.transform(non_nan_values)
            
    return features_df, encoders

def impute_missing_values(features, target_df, target_col):
    # ensure all features are encoded
    features, encoders = encode_features(features.copy())
    
    # split data based on NaN values in target
    known_idx = target_df[target_col].dropna().index
    unknown_idx = target_df[target_col].isna().index
    
    X_train = features.loc[known_idx]
    y_train = target_df.loc[known_idx, target_col]
    
    X_test = features.loc[unknown_idx]
    
    # train Decision Tree model
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    
    # predict NaN values
    predicted_values = model.predict(X_test)
    
    # replace NaN values in the target DataFrame
    target_df.loc[unknown_idx, target_col] = predicted_values
    
    return target_df

# apply the imputation function for each target DataFrame
# impute for job and update the feature set
job_df = impute_missing_values(new_df_features, job_df, 'job')
new_df_features['job'] = job_df['job']

# impute for education and update the feature set
education_df = impute_missing_values(new_df_features, education_df, 'education')
new_df_features['education'] = education_df['education']

# impute for contact
contact_df = impute_missing_values(new_df_features, contact_df, 'contact')
new_df_features['contact'] = contact_df['contact']

# add back NaN to unknown values
new_df_features.replace('unknown', np.nan, inplace=True)
new_df_features.isna().sum()

# predictions still ends up in NaN for contact because of the lack of representation of factors levels for this variable
new_df_features.drop('contact', axis=1, inplace=True)

# quality check
new_df_features.head()

# merge with label dataset so we can finish removing any other NaN still left after imputation
merged_df = pd.concat([label, new_df_features], axis=1)  # axis=1 for horizontal merge

# quality check
merged_df.head()

# drop any remaining NaN now with the full dataset again
merged_df.dropna(inplace=True) 
# 36 NaN removed. This may vary slightly every time we do ML imputation. 
# These NaN were in the variable education. Perhaps using another algorithm 
# other than DecisionTree we can overcome this issue, but given the correlation 
# of these variables with y and the imputation here just for training, there was no need to worry about it now

# perform random under-sampling treatment to correct for imbalanced class problem in the dataset
# class count
class_count_0, class_count_1 = merged_df['y'].value_counts()

# Separate class
class_0 = merged_df[merged_df['y'] == 0]
class_1 = merged_df[merged_df['y'] == 1]# print the shape of the class
print('y 0:', class_0.shape)
print('y 1:', class_1.shape)

# apply subsampling
class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1], axis=0)
print("total class of 1 and0:",test_under['y'].value_counts())# plot the count after under-sampeling
test_under['y'].value_counts().plot(kind='bar', title='count (target)')

# preparing for ML prediction, split in label and features
label = test_under.loc[:, ['y']]
new_df_features = test_under.drop('y', axis=1)

# define columns to be label encoded
columns_to_encode = ['marital', 'default', 'housing', 'loan', 'month', 'job', 'education']
for col in columns_to_encode:
    encoder = LabelEncoder()
    new_df_features[col] = encoder.fit_transform(new_df_features[col])

# Define columns to be scaled
columns_to_scale = ['age', 'balance', 'day', 'duration', 'campaign']
for col in columns_to_scale:
    scaler = StandardScaler()
    new_df_features[col] = scaler.fit_transform(new_df_features[col].values.reshape(-1, 1))

# drop features because of consistent lack of contribution to the prediction - prediction also improves without it
features.drop(['default', 'loan'], axis=1, inplace=True)
    
# splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(new_df_features, 
                                                    label, 
                                                    test_size=0.2, 
                                                    random_state=0)

model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train,y_train)

# predict in the validation dataset
y_pred = model.predict(X_test)

# classification report (includes f1-score, predicision, recall, and accuracy)
target_names = ['No', 'Yes']
print(classification_report(y_test, y_pred, target_names=target_names))

# f1-score only for the purpose of the assignment
print("Training Score is {} and Testing Score is {}".format(
    f1_score(y_train, model.predict(X_train)),
    f1_score(y_test, model.predict(X_test))
))

# extract confusion matrix
confusion_matrix(y_test, y_pred)

# feature importances
importances = model.feature_importances_
print("Feature importances:", importances)