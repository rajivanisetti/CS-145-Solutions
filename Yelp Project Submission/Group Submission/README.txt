Preprocessing
----------------------------------------------------------------------------
1. Setup
	a. import pandas and numpy
1. Preprocessing Business Data
	a. Read in the business data from the csv file
	b. Set up default values for ambience, and prepare to remove NaN values
	c. Feature Selection:
		i. Remove features using .drop() function
	d. Reformattng
		i. Use view_column_valueus, expand_dict_to_columns, and replace_column_nan
		   to view the dataframe, expand the dictionaries, and replace NaN values
		ii. Replace all NaN values to the default values set up in part b)
		iii. Change all categorical data to numerical
	e. Prune it one more time by removing more irrelevant features (>50% NaN values)

2. Preprocessing User Data
	a. Read in the user data from the csv file
	a. Drop features that do no intuitively relate to rating
	b. Check that all data is numerical, and that there are no NaN values

3. Preprocessing Review Data
	a. Read in the review data from the csv file

4. Formatting Training Data
	a. Merge the user and business data according to the user-id and business-id 
	   relationships from the review data. For example, if one row in reviews
	   has user-id: 1 and business-id: 1, then the corresponding row in the 
	   training data has the features of user '1' and business '1'. 
	b. Normalize the useful feature by dividing it by the total number of reviews
	   the corresponding user has given. 


Training
----------------------------------------------------------------------
1. Setup
	a. Import the four models from the sklearn library: 
		KNeighborsClassifier, RandomForestClassifier, 
		MLPClassifier, and MLPRegressor
	b. Import some verification tools: 
		classification_report, confusion_matrix, accuracy_score
		make_scorer, mean_squared_error
	c. Define the verification function to evaluate for all of the models' predictions
				
2. KNN Training Model
	a. Use the KNeighborsClassifier function to run KNN, and print out the
	   accuracy and MSE of the training and validation report
	b. Repeat a. for 10 different k values

3. Random Forest Training Model
	a. Use the RandomForestClassifier function to run Random Forest, and print 
	   out the accuracy and MSE of the training and validation report
	b. Repeat a. for 3 different depth values

4. MLP Classification Training Model
	a. Use the MLPClassifier function to run KNN, and print out the
	   accuracy and MSE of the training and validation report
	b. Repeat a. for all combinations of:
		3 different max_iter values
		3 different alph values.

5. MLP Regressor Training Model
	a. Use the KNeighborsClassifier function to run KNN, and print out the
	   accuracy and MSE of the training and validation report
	b. Repeat a. for all combinations of:
		3 different max_itr values
		3 different alph values
		2 different hidden_layer_size values


Best Model
-------------------------------------------------------------------------
1. Run a training and validation report on the best model selected from above
	a. Once with normal dataset
	b. Once with combined dataset
2. Get the full parameters of the model (for science)
3. Predictions
	a. Read in the test_queries.csv file as the test dataset
	b. Process the test dataset, and adjust the shape accordingly
	c. Feed the data into the model, and take the output as submit_y
	d. Convert the output into submission format, and feed it into a csv file