# Chicago Crime Prediction
-----
The objective is to predict the number of crimes for all types of crimes listed in the [City of Chicago](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) dataset.


## Getting Started
-----
To reproduce our result, start by cloning this repo.
```bash
git clone https://github.com/Ravisutha/CrimePrediction.git
```

### Prerequisites
----
Following are some packages that are a must to run the code.
>1. [networkx](https://networkx.github.io/)  - To build the graph and to find similar communities.
>2. [sklearn](http://scikit-learn.org/stable/) - To predict number of cirmes.
>3. [pandas](https://pandas.pydata.org/) - To handle data.

If you use `anaconda`, it would be simpler to use the environment `crime_predict.yml` which can be found in the root directory of this repo.
```bash
conda env create -f crime_predict.yml
```

### Run prediction models
----
The [Code](https://github.com/Ravisutha/CrimePrediction/tree/master/Code) directory contains code for handling dataset, creating network using networkx, analyzing the data (making predictions) and for visualizing the results. These are categorized into appropriate folders. To predict the number of crime for a given period, navigate to [`Analysis`](https://github.com/Ravisutha/CrimePrediction/tree/master/Code/Analysis) directory and run the `predict.py` code.

```bash
cd ./Code/Analysis
python predict.py
```
  You can check the predicted outputs in [`./Data/Total_Data/Output`](https://github.com/Ravisutha/CrimePrediction/tree/master/Data/Total_Data/Output). To visualize the output:
```bash
cd ./Code/Visualize
jupyter notebook boxplot_fullcrime.ipynb
```

### Authors
-----
* [Dr. Ilya Safro](https://people.cs.clemson.edu/~isafro/)
* [Ravisutha Sakrepatna Srinivasamurthy](https://www.linkedin.com/in/ravisutha/)
* [Saroj Dash](https://www.linkedin.com/in/saroj31/)
