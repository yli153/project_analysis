from pandas import read_csv
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# recursive feature elimination 
dataframe = read_csv(r"C:\Users\sli98\OneDrive\Desktop\cs1951a\final_project_regression\data_with_name.csv")
X = dataframe.drop(['is_winner', 'row', 'name'], axis=1)
y = dataframe['is_winner']
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 6)
fit = rfe.fit(X.values, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# The 6 most important features are serve_win_pct_gps, pct_win_long_rallies, 
# pct_pts_won_af_return, pct_pts_won_fh_return, pct_pts_won_bh_return, pct_pts_won_sv_in_leq_3s_serve




