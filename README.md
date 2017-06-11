# NewsXtract

## Purpose

The aim of this project is to identify correlations in news coverage using matrix factorisation techniques.

## Theory

The main theory for this project is based around [recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) based on Matrix Factorisation, and [Bayesian Personalised Ranking](https://arxiv.org/abs/1205.2618).

## Results

The main metric we will use to validate our system is an [AUC score](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it). Figure 1 shows the convergence of the AUC score for cross-validated parameters $\alpha = 0.1$ (the learning rate), $\lambda = 0.01$ (the regularization factor), and $K=20$ (the number of latent factors).

![Figure 1](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/auc.png)
__*Figure 1* : AUC score for cross-validated parameters__

## Examples
### Plotting the sources' latent space

Once matrix factorizations have been computed, we have got a representation of both sources and events in the latent space (of size `N_sources x K` and `K x N_events` respectively). We feed this higher dimensional representation to [t-SNE](https://lvdmaaten.github.io/tsne/) to obtain a 2-D (i.e. plotable) space, in the case of sources as this is the most interesting space to search for correlations in.

![Figure 2](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/top20clusters.png)
__*Figure 2* : Sources' latent space__



### Clustering in the sources' latent space

Figure 2 shows the emergence of some distinct clusters, but this is a visual interpretation. We will run DBSCAN on the data to show the existence of these clusters.

![Figure 3](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/dbscan.png)
__*Figure 3* : DBSCAN run on the sources' latent space__

### Identifying interesting clusters

If we assume that the clusters reveal some correlation between their neighbours, several interesting patterns emerge from this representation.

#### Geographic clustering

One obvious source of correlation would be geographic proximity between the sources : news sources in similar locations should often talk about similar, local, issues. this can be shown in the following figures.

![Figure 4](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/tsne-zoom2.png)
__*Figure 4* : An Australia - New Zealand cluster__

![Figure 5](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/tsne-zoom3.png)
__*Figure 5* : A UK cluster__

#### Revealing larger structures

An interesting result of our study was the emergence of clusters that had seemingly nothing to do with each other. However, after a little bit of digging, we discovered that these sources were part of larger entities, which dispatch global / national news, leaving these smaller sources deal with their local news independently. 


![Figure 6](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/tsne-zoom6.png)
__*Figure 6* : A cluster of news groups lead by CNN__

Note that they are also all powered by the same Content Management System (called Lakana).

#### Medium based clusters

![Figure 7](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/tsne-zoom1.png)
__*Figure 7* : A cluster of American Radio stations__

Interestingly this cluster joins many radio that have organised into a group headed by NPR, the Public Radio Initiative, the BBC and American Public Radio.

### Dependence on News agencies

The news groups that we indentified in the previous section can also be found on a larger scale, with many sources depending on news-wires for a lot of their reporting. Two main players dominate this area : *Reuters* and the *Associated Press*. We can show each news source's dependance on these sources by showing the distance between their two coverages.

![Figure 8](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/ap_dist.png)
__*Figure 8* : Log-Distance of each news source to the Associated Press wire__

![Figure 9](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/reuters_dist.png)
__*Figure 9* : Log-Distance of each news source to the Reuters wire__


### Predicting coverage

The final step is to see how good our model is at predicting a source's coverage of an event. We will rank the predictions made by our model for the holdout event (the one we used as a test) and see where it ranks in the list of predicted events. 

![Figure 10](https://github.com/JRappaz/NewsXtract/blob/gdelt/BPR-example/img/hist_ranking.png)
__*Figure 10* : Distribution of the ranking counts for the holdout events__

As we can see the model is in general quite good at predicting if an event was covered or not by a source (this is shown by the fact that most of the events predicted ranked quite high).
