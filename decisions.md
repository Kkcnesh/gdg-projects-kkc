hello jugde below is all the decisions which have led to this questionable ml model which may or may not predict your future accurately 🤓

1- The data set misery : the original data set provided was of 10,000 entries with multiple columns that had to be cleaned and used to train the model basic steps before training a model is to provide a good data set. the data set had approx 9000 duplicate entries making most of it just noise so the actual data to be used for training was actually a 1000 rows 
also some parameters had no business being used for training such as name age opinion which had to be removed.

2- model performance: being honest the model is pretty bad like v v bad going according to industry standards but there is a reason for this THE DATA SET poor entries as most of the stats had no correlation with being placed or unplaced a student with 0 projects 0 skills was marked as placed while a highly talented student wasn't this wasn't a unique case this happened across multiple instances contributing to the poor training of the model. Instead of changing the target labels, artificially creating correlations, or repeatedly tuning the models only to increase the score, the observed results were retained.The purpose of the project is to demonstrate a complete and honest machine learning pipeline.
Increasing the score artificially would not represent genuine predictive performance.The weak results also became an important finding about the supplied synthetic dataset.

3-data processing: most parameters such as languages or certifications can't be fed directly to models as data they have to make sense to it so i just used no. of languages ex: if a student knows c,java,python so no. of languages = 3 similar decision was made for certifications too. 

4- training: We tested three models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

After comparing them, Random Forest performed best overall on our held-out test set.
So we chose Random Forest as the model that powers the actual Streamlit prediction.
among the models we tested, Random Forest gave the strongest overall results

5-A simple Streamlit interface was added to allow users to enter a student’s information and receive a prediction.