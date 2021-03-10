# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this research project, I have applied two different methods to get the best classification model given the same dataset.
First, I have used a customised training script using Scikit-learn, optimizing the hyperparameters with Hyperdrive.
After that, I have run AutoML providing as input the same dataset, without specifying the algorithm or parameters to be 
used, so that AutoML itself could figure them out.

After running both experiments, the result observed is that both mechanisms return models with very similar accuracy
(0.9150 with Hyperdrive, and 0.9154 with AutoML), with less programmatic effort and execution time required to run the 
experiment using AutoML.

## Scikit-learn Pipeline
The data used for the pipeline is a Bank Marketing dataset used to obtain a classification model.
The data contains several features, some are numerical, and some are categorical. The last ones are 
encoded before the training starts.  

The pipeline defined in this experiment consists in a training script built using Scikit-learn. In this script, data is
read and cleaned first. Afterwards, a Logistic Regression algorithm is applied to part of the data (training set) with the 
hyperparameters passed to the script through arguments. After the training is completed and the model is evaluated, the 
obtained accuracy is logged into the run object.

With Hyperdrive, you can run this process as many times as you want passing different combinations of hyperparamaters
defined in the Hyperdrive configuration object.  

**What are the benefits of the parameter sampler you chose?**

The random parameter sampler is suitable when you don't have yet a very well defined space of hyperparameters to explore.
It is a good approach if you want to do an initial exploration of hyperparameters, and then refine the search space based 
on the results obtained with it. 

**What are the benefits of the early stopping policy you chose?**

Bandit termination policy is an appropriate termination policy when you consider that the model is not going to be able 
to improve the accuracy beyond a small margin. For that reason, you can define that margin (in this experiment, it was used 
a value of 10%) that you consider enough to terminate the run. 

## AutoML
In the experiment built with AutoML, 21 runs with different algorithms and hyperparameters were executed, obtaining a
different model from each of them. The model that obtained the highest accuracy was using VotingEnsemble as algorithm. 
Some of the hyperparameters found to be the best are: solver=newton-cg, max_iter=100 and l2 penalty.

## Pipeline comparison

Both experiments provide a relatively similar accuracy (around 91.50%). However, in terms of simplicity of the pipeline, 
with AutoML you don't have to worry about providing the algorithm. You just have to prepare the data and send it to AutoML,
and it will look for the best algorithm and the best hyperparameters. In term of performance, while the experiment run with
scikit-learn took 1.5 min, with automl was only 0.5 min. This was due to the fact that the algorithm found by automl is 
more optimal than LogisticRegression. 

## Future work

In the case of the scikit-learn experiment, using different algorithms may result on more accuracy and better performance. 
Also, refining a bit the hyperparemeters after doing the random sampler approximation may impact the performance and accuracy as well.
In the case of the AutoML, you could try with higher timeouts of the experiments, so that AutoML can continue improving 
the model.
