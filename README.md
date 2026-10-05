# spotter-app
mchine learning 
r2 accurcy 87%
compare many models 
gets best results
Freight Rate Prediction

This project was built as part of a machine learning assessment for a freight rate prediction problem.

The goal was to predict the expected freight rate for transportation loads.

I started by loading the training data and inspecting the available columns.

The first step was understanding the target and the features that could affect the freight rate.

I checked the data types and looked for missing or unusual values.

After preparing the data, I separated the target variable from the input features.

I then created a validation setup to evaluate the models.

I started with a simple regression approach to establish a baseline.

After that, I tested different machine learning models.

I compared their predictions using regression metrics.

The main metrics I focused on were R², MAE, and RMSE.

One of the strongest models in my experiments was Gradient Boosting.

The best result I achieved was approximately:

R²: 0.8618

MAE: 126.39

RMSE: 543.59

I did not stop after getting the first reasonable result.

I compared different approaches and looked at how the changes affected the validation performance.

The project also required generating predictions for a separate December dataset.

I created the required prediction output and checked that the load IDs matched the expected format.

Another part of the assessment was generating a chart from the December predictions.

I also organized the project so that the main steps could be reproduced from the repository.

This project was especially useful because it was based on a real business-style machine learning problem rather than only a tutorial dataset.

It gave me experience working with a defined prediction task, validation requirements, output files, and model comparison.

It also showed me the importance of making a machine learning project reproducible and easy to evaluate.

I used

Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

Future Improvements

I would like to improve the feature engineering, test additional boosting algorithms, perform hyperparameter tuning, and analyze prediction errors in more detail.

I would also like to turn the final model into a simple API for making freight rate predictions.
