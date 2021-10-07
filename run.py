## Script for making final prediction.
import pickle
import data_prep
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import data_prep

np.random.seed(0)

def main():
    model1_drop = ['C24', 'C19', 'C14', 'C28']
    model2_drop = ['C23', 'C19', 'C14', 'C27', 'C28']

    # Get models
    with open('./model1-comnbayes.pickle', 'rb') as f:
        grid_search_1 = pickle.load(f)

    with open('./model2-svc.pickle', 'rb') as f:
        grid_search_2 = pickle.load(f)
    
    model1 = grid_search_1.best_estimator_    
    model2 = grid_search_2.best_estimator_    

    # Get data
    df1_train, df1_test = data_prep.get_prepped_dataset(bins=5, normalize=False)
    df2_train, df2_test = data_prep.get_prepped_dataset(bins=None, normalize=False)
    
    # Drop cols
    df1_train = df1_train.drop(model1_drop, axis=1)
    df1_test = df1_test.drop(model1_drop, axis=1)
    df2_train = df2_train.drop(model2_drop, axis=1)
    df2_test = df2_test.drop(model2_drop, axis=1)

    # Prep X and y
    X1_train = df1_train.drop('Class', axis=1)
    y1_train = df1_train['Class']
    y1_train = LabelEncoder().fit_transform(y1_train)
    X1_test = df1_test.drop('Class', axis=1)
    X2_train = df2_train.drop('Class', axis=1)
    y2_train = df2_train['Class']
    y2_train = LabelEncoder().fit_transform(y2_train)
    X2_test = df2_test.drop('Class', axis=1)

    # Sanity Check
    print("Model 1 (CNBayes) sanity check:")
    print(f"Accuracy: {np.average(cross_val_score(model1, X1_train, y1_train , cv=10))}")
    print(f"F1 Score: {np.average(cross_val_score(model1, X1_train, y1_train , cv=10, scoring='f1_macro'))}")
    
    print("Model 2 (SVC) sanity check:")
    print(f"Accuracy: {np.average(cross_val_score(model2, X2_train, y2_train , cv=10))}")
    print(f"F1 Score: {np.average(cross_val_score(model2, X2_train, y2_train , cv=10, scoring='f1_macro'))}")

    # Train on whole test set
    print("Training...")
    model1.fit(X1_train, y1_train)
    model2.fit(X2_train, y2_train)

    # Make Predictions
    print("Making predictions...")
    prediction1 = model1.predict(X1_test)
    prediction2 = model2.predict(X2_test)

    # Save to file
    y_test_id = list(pd.read_csv("./data2021.student.csv")[1000:]['ID'])
    
    assert len(y_test_id) == len(prediction1)
    assert len(prediction1) == len(prediction2)

    final_output = []
    for ii in range(len(prediction1)):
        final_output.append([y_test_id[ii], int(prediction1[ii]), int(prediction2[ii])])

    final_output_table = pd.DataFrame(data=final_output, columns=["ID", "Predict 1", "Predict 2"])
    final_output_table.to_csv("predict.csv", index=False)

    print("Done!")

if __name__ == '__main__':
    main()