clean.py => cleaning the data

traindataset.csv => cleaned data csv file

model.py => training file

--RESULT---
Algorithm => GradientBoostingRegressor
Root Mean Square error => 55745.56


==========second attempt========

New additions:

	Did feature scalling

	Removed rows that have ouliers in the target variable 

	Changed the model


------RESULT----

Model = > ExtraTreesRegressor

Root Mean Square error => 34878.80

Note : Please try using sklearn.ensemble.HistGradientBoostingRegressor; i am not being able to update my scikit-learn for some reason. 



==========third attempt========

New addition:
	 
	used EllipticEnvelope for outlier detection [95% of the data is outliers]



------RESULT----

Model = > ExtraTreesRegressor

Root Mean Square error => 3806.73

