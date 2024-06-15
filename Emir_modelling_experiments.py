import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

my_path = 'C:/Users/ameer/Desktop/AI/ML4QS/BackUp/Emir_Datasets'

full_data = pd.read_parquet(my_path+"/full_dataset.parquet.gzip")#full_data["SM_platform"] = full_data["source_name"].str.extract(r'([A-Za-z]+)')

random.seed(51)
testing_categories = [1,2,3,4,5,6]
random.shuffle(testing_categories)

testing_set = [f"{platform}{number}" for platform, number in zip(full_data["SM_platform"].unique(), testing_categories)]
testing_df = full_data[full_data["source_name"].isin(testing_set)]
training_df = full_data[~full_data["source_name"].isin(testing_set)]

#only select standardized columns
columns_to_select = [col for col in full_data.columns if 'zscore' in col]+["SM_platform", "index"]
training_df = training_df.loc[:,columns_to_select]
testing_df = testing_df.loc[:,columns_to_select]
param_grid = { 
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 

random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid)

random_search.fit(training_df.drop(columns=["SM_platform"]), training_df["SM_platform"])

'''
The best parameter set is supposed to be this (but it serves as an inspiration)

bootstrap=True, ccp_alpha=0.0, class_weight=None,criterion='gini', max_depth=6, 
max_features='log2', max_leaf_nodes=9, max_samples=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=50,
n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False
'''

model_grid = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) 

model_grid.fit(training_df.drop(columns=["SM_platform"]), training_df["SM_platform"])
y_pred_grid = model_grid.predict(testing_df.drop(columns=["SM_platform"]))
print(classification_report(y_pred_grid, testing_df["SM_platform"])) 
print(confusion_matrix(y_pred_grid, testing_df["SM_platform"]))
print(accuracy_score(y_pred_grid, testing_df["SM_platform"]))

importances_gsmodel = model_grid.feature_importances_
feature_names_gsmodel = training_df.drop(columns=["SM_platform"]).columns

# Create a DataFrame for visualization
feature_importance_df_gsmodel = pd.DataFrame({
    'Feature': feature_names_gsmodel,
    'Importance': importances_gsmodel
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df_gsmodel.head())

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df_gsmodel['Feature'], feature_importance_df_gsmodel['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()
plt.show()