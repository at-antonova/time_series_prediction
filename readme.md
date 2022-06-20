## Chaotic time series prediction: self-healing algortihm for prediction with clustering

### Abstarct
The paper discusses a further development of predictive clustering approach for multi-step-ahead prediction of chaotic time series. According to this approach for each point to be predicted a large set of possible predicted values can be calculated. The algorithm then evaluates whether a point is predictable or non-predictable and calculates unified prediction for those points which were found predictable. 
A novel strategy of the self-healing algorithm is suggested in the present paper. The self-healing algorithm is an iterative algorithm which takes predictions obtained by the base prediction algorithm and iteratively tries to "heal" the part of the time series to be predicted by finding new possible predicted values, updating the status of points from predictable to non-predictable or backwards and calculating new unified predictions for predictable points.
In the present paper a number of novel algorithms of identifying non-predictable points and algorithms of calculating unified prediction are suggested. Experiments for hyperparameters selection for the self-healing algorithm and comparison with base prediction algorithm were conducted. **They showed that for the Lorenz time series average prediction errors increase slightly, however at the same time the number of non-predictable points decreases drastically compared to the base prediction algorithm.**

### Files overview

#### code/ overview
*code/* - source code in Jupyter Notebooks and *time_series_prediction* Python module with prediction algorithm, experiments and graph generation

module time_series_prediction.predictor contains classes:
- *TimeSeriesPredictor* - main class for time series prediction workflow
- *NonPredModel* and its inherits - classes for algorithms of identifying non-predictable points

module time_series_prediction.experiment contains functions of perfomed numerical experiments

module time_series_prediction.graphs contains functions of main graphs generation that were provided in report

Details of implementation are given in comments in code.

#### Usage

Example of usage provided in example.py file. Predicting Lorenz time series. Base prediction uses algorithm of identifying non-predictable points of limit cluster size, gamma=0.5, min_clusters=2, algorithm of calculating unified prediction - mean of the largest cluster DBSCAN. Self-healing algorithm uses weird patterns with base non-predictable points algorithm of limit cluster size, gamma=0.5, min_clusters=2. Unified prediction - mean of largest weighted cluster DBSCAN, method - factor with factor=0.7.

#### data/ overview
*data/* - clustered motifs and time series data

lorenz.txt -- not my file
lorenz.dat -- generated by me, main source of lorenz time series

thrown_points_100.dat -- specific thrown points data for thrown points experiment consistency

electricity_Y1.dat -- training time series data for electricity load
electricity_Y2.dat -- test time series data for electricity load

**motifs/**
motifs_db_b20.dat -- Lorenz, DBSCAN motifs, beta=20%
motifs_db_b100.dat -- Lorenz, DBSCAN motifs, beta=100%
motifs_wi_b20.dat -- Lorenz, Wishart motifs, beta=20%
motifs_wi_b100.dat -- Lorenz, Wishart motifs, beta=100%
motifs_electricity_db_b20.dat -- electricity load, DBSCAN motifs, beta=20%

**models/**
lr.dat -- pre-trained logistic regression NonPredModel

#### docs/ overview
*docs/* - final paper and applications, presentation (in Russian)

#### results/ overview
*results/* - main tables with final results, source files (.dat) for them