import numpy as np
import pandas as pd
from csv import reader
import random
import csv
import statsmodels.api as sm
from statsmodels.tools import eval_measures

# splitting data into [prob, 1-prob]
def split_data(data, prob):
    """input: 
     data: a list of pairs of x,y values
     prob: the fraction of the dataset that will be testing data, typically prob=0.2
     output:
     two lists with training data pairs and testing data pairs 
    """
    length = len(data)
    rand = [random.random() for i in range(length)]
    test = []
    train = []
    for i in range(length):
        if rand[i] <= prob:
            test.append(data[i])
        else:
            train.append(data[i])
    return train, test


# splitting data into training and testing sets 
def train_test_split(x, y, test_pct):
    """input:
    x: list of x values, y: list of independent values, test_pct: percentage of the data that is testing data=0.2.

    output: x_train, x_test, y_train, y_test lists
    """
    zipped = zip(x, y)
    train, test = split_data(list(zipped), test_pct)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for (x, y) in train:
        x_train.append(x)
        y_train.append(y)

    for (x, y) in test:
        x_test.append(x)
        y_test.append(y)

    return x_train, x_test, y_train, y_test


# main methods to run linear regression
if __name__=='__main__':

    # DO not change this seed. It guarantees that all students perform the same train and test split
    random.seed(1)
    # Setting p to 0.2 allows for a 80% training and 20% test split
    p = 0.2

    def load_file(file_path):
        """input: file_path: the path to the data file
           output: X: array of independent variables values, y: array of the dependent variable values
        """
        data_df = pd.read_csv(file_path)
        # X = data_df[['win_pct_bps', 'win_pct_deuce', 'serve_win_pct_gps', 'pct_ufcd_err', 'pct_opponent_fcd_err', \
        #              'pct_win_long_rallies', 'pct_win_long_rallies_initiated_by_opponent', 'pct_unfcd_error_rally', \
        #              'pct_fcd_error_pts_rally', 'pct_pts_won_af_return', 'pct_unreturnable_serve', 'pct_fcd_err_serve', \
        #              'pct_pts_won_sv_in_leq_3s_serve']]
        X = data_df.drop(['is_winner', 'name', 'row'], axis=1)
        y = data_df['is_winner']
        return X , y

        
    # loading file and running linear regression on training data 
    X, y = load_file(r"C:\Users\sli98\OneDrive\Desktop\cs1951a\final_project_regression\data_with_name.csv")
    X = sm.add_constant(X)
    x_train, x_test, y_train, y_test = train_test_split(X.values, y, p)
    model = sm.OLS(y_train, x_train)
    results = model.fit()
    print(results.summary())
    # applying the regression results to the testing data 
    train_y_cap = results.predict(x_train)
    y_cap = results.predict(x_test)
    training_MSE = eval_measures.mse(y_train, train_y_cap)
    testing_MSE = eval_measures.mse(y_test, y_cap)
    print('training r-squared: '+ str(results.rsquared))
    print('training MSE: '+str(training_MSE))
    print('testing MSE: '+str(testing_MSE))

    # using the coefficients and constant from regression to predict
    coefficients = np.array(results.params)
    player_names = []
    prediction_inputs = []
    with open(r"C:\Users\sli98\OneDrive\Desktop\cs1951a\final_project_regression\data_with_avg.csv", 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        if header != None:
            for row in csv_reader:
                player_names.append(row[0])
                player_data = [1]
                player_data.extend(row[2:])
                prediction_inputs.append(player_data)
    # computing the predicted likelihood of winning for each player
    inputs = np.array(prediction_inputs).astype(float)
    prediction_outcome = np.zeros(len(player_names))
    for i in range(len(player_names)):
        prediction_outcome[i] = np.dot(coefficients, inputs[i].T)
    # sorting the results from maximum to minimum 
    sorted_indices = np.argsort(prediction_outcome)[::-1]
    sorted_players = []
    player_prob = []
    for i in sorted_indices:
        player_prob.append((player_names[i], prediction_outcome[i]))
    info = np.array(player_prob)
    # writing the prediction results to a csv file 
    df = pd.read_csv(r"C:\Users\sli98\OneDrive\Desktop\cs1951a\final_project_regression\prediction.csv")
    df['Name'] = info[:, 0]
    df['Result'] = info[:, 1]
    df.to_csv(r"C:\Users\sli98\OneDrive\Desktop\cs1951a\final_project_regression\prediction.csv")
    
   
    

