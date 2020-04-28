# Item-Underrecommendation-Bias
Code for the SIGIR20 paper -- Measuring and Mitigating Item Under-Recommendation Bias inPersonalized Ranking Systems


## Data
ml1m-2, yelp-2, and amazon-2 are the three datasets with two sensitive groups. ml1m-6, yelp-4, and amazon-4 are the three datasets with multiple sensitive groups. There is no original data files in this repo, if you want to get the original data files, please refer to the paper to see the original sources of these datasets.

## Requirments
python 2  
tensorflow 1.13.0  
numpy  
sklearn  
pandas  
matplotlib

## Excution
Run DPR_RSP.py to run DPR-RSP model, run DPR_REO.py to run DPR-REO model, and run BPR.py to run BPR model. All the hyperparameters are described in the paper.

