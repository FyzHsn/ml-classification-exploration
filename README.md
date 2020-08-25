# A Broad Exploration of ML Techniques
Machine learning, as a field, has been evolving rapidly. While the popularity of packages such as scikit-learn has been stable, distributed and cloud computing software such as Apache Spark has become crucial for the handling of big data and training complex models. Part of my motivation behind the work presented here is to develop a broad ML tech stack. The main themes are:

1. scikit-learn: stochastic gradient descent, logistic regression, imblearn, Google Compute Engine
2. PySpark: MlLib, Google Cloud Compute Engine, Dataproc
3. PyTorch
4. TensorFlow
5. Julia 

## Dataset
The dataset used for experimentation here has the following properties:
1. There are 3,738,937 rows.
2. There are 16 features including the target feature `install`. 
3. There are 7 categorical features. Two of these features `campaignId` and
`sourceGameId` have cardinality of 9692 and 34,849. This means that One Hot
Encoding necessarily leads to the addition of a large number of feature
vectors. And yet, for the purpose of testing the performance of the tech
stack, we do not attempt to reduce the cardinality of the categorical
features.
4. The remaining 9 features are either numeric of datetimes.
5. The two datetime features will be dropped in favour of a combined numeric
feature.

While this is not an adequate amount of details for the purpose of
understanding the modeling choices, it is enough to understand the time and
memory complexity of the problem (and how each member of the tech stack
performs comparatively).  
    

## Model & Package Performance
### Scikit-learn + Imbalanced-learn deployed on Compute Engine
Compute Engine Specs:
- n1-standard-4 (4 vCPUs, 15 GB memory) 
- Debian GNU/Linux 9

Logistic Regression training took 53 seconds with precision 2%, recall 67% and area under roc of 0.72.
Stochastic Gradient Descent training took 13 seconds with 2%, recall 52% and area under roc of 0.69.

