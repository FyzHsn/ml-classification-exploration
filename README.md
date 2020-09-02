# A Broad Exploration of ML Techniques
Machine learning, as a field, has been evolving rapidly. While the popularity of packages such as scikit-learn has been stable, distributed and cloud computing software such as Apache Spark has become crucial for the handling of big data and training complex models. Part of my motivation behind the work presented here is to develop a broad ML tech stack. The main themes are:

1. scikit-learn: stochastic gradient descent, logistic regression, imblearn, Google Compute Engine
2. PySpark: MlLib, Google Cloud Compute Engine, Dataproc
3. PyTorch
4. TensorFlow
5. Julia 

The time comparison to perform similar tasks using different tech stack is
shown below.
![](https://github.com/FyzHsn/ml-classification-exploration/blob/develop/images/performance.png?raw=true)

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
4. The remaining 9 features are either numeric, datetimes or binary.
5. The two datetime features will be dropped in favour of a combined numeric
feature.

While this is not an adequate amount of details for the purpose of
understanding the modeling choices, it is enough to understand the time and
memory complexity of the problem (and how each member of the tech stack
performs comparatively).  
    

## Model & Package Performance
### 1. Scikit-learn + Imbalanced-learn deployed on Compute Engine
Compute Engine Specs:
- n1-standard-4 (4 vCPUs, 15 GB memory) 
- Debian GNU/Linux 9

Comparison of optimal model performances:
* Logistic Regression training took 53 seconds with precision 2.4%, recall 67.8%
and area under roc of 0.73.  
* Stochastic Gradient Descent training took 13 seconds with 2.3%, recall 53%
and area under roc of 0.69. 

### 2. Scikit-learn + Imbalanced-learn deployed on Macbook Air
Macbook Air Specs:
- Processor 1.6 GHz Dual-Core Intel Core i5
- Memory 8 GB 2133 MHz LPDDR3

Comparison of optimal model performances
* Logistic Regression training took 36 seconds which is actually faster than
 on GCP.
* Stochastic Gradient Descent took 15 seconds which is comparable to GCP.

## 3. PySpark + Google Cloud DataProc
Grid search took 38 minutes, while model training took 0.7 seconds with
precision 2.5%, recall 63% and area under roc of 0.74. The cluster was
comprised of 4 worker nodes and 1 master node. Each (n1-standard) node is
comprised of 4 vCPUs, 15 GB memory.

 
 

