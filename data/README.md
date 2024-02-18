# ML Project - Pairs Trading via Unsupervised Learning - Replicating Han(2022) results
I used data from professor Dacheng Xiu's web-site (https://dachxiu.chicagobooth.edu/download/datashare.zip ) . Data are similar ( just 2 parameters were missing - IPO and sue ) so I did analysis with 76 parameters instead 78.

Since there were no details about missing values and handling outliers I took max 2 missing values ( out of 76 ) and I filled NA with the median values.

I did not handle outliers since I did not want to generate some potential artificial point of similarity for clustering algorithms later.

I did not have adjusted returns (  CRSP - Center for Research in Security Prices ) so I took momentum as return ( since mom is actually adj. return ).  

My results were not as good as in paper, especially for short legs. 


#Project goal was to replicate results from the Han(2022) paper i.e to evaluate Pairs Trading strategy and not to extend topics ( like stationarity check, optimal cluster number, optimal PCA components  ...). Due to a lack of time, robustness and systematic risk factors evaluations of models ( using factor regressions - FF3,FF5, q5 ) are not done.

Strategy is based on using PCA for momentum and firm characteristics for preprocessing data and then using Unsupervised Learning ( clustering ) for detecting similar companies as candidates for pair trading. 
Methodology:
In the initial step ( ML Project Pairs Trading - Initial EDA.ipynb ) data set was examined. Data source is from DachengXiu website updated  in June 2021. Data set covers the time 1957 - 2021 and consists of 95 company characteristics of 1000 - 8000 companies. Visualisation of missing values shows expected behavior that significant amounts of data are missing at the beginning of the dataset but are quite ok for the period of our interest ( 1976 - 2020 ). Visualisation of duplicates shows the stale nature of company characteristics since most data are quarterly published. Han discarded company characteristics with more than 15% missing values and got a list of 78. I made the same conclusion except that in my data set, 2 characteristics were missing ( IPO and sue ) so I continue the analysis with 76. ( result of this intermediate step - df_radni.csv )

In the EDA part ( ML Project Pairs Trading - EDA..ipynb ) I dropped data with missing at least one mom1m in 48 last months and with missing values in any of 76 characteristics. Since there were no details about missing values and handling outliers,and I needed to compare clustering results with Table 1 from the paper, I did inverse engineering and concluded that max 2 missing values ( out of 76 ) will give approximately the same number of companies in the period of 1990 - 2020. After data cleaning with these assumptions, one characteristic ( tb - Tax income to book income ) popped up to be most frequented ( up to 16% ). Others were negligible. I filled NA with the median value of tb on that date. During analysis, I noticed a drop of quality of data ( more missing values ) on every end of Aug. I did not handle outliers since I did not want to generate some potential artificial point of similarity for clustering algorithms later.
( result of this intermediate step - df_preprocessing_done.csv )

In the Normalization and PCA part ( ML Project Pairs Trading - Normalization and PCA.ipynb ) I normalize data 76 company and 48 momentum characterics ( standard scaling ) and do PCA. Calculating and 90%,95% and 99% variation explainability. In paper 99% is used. ( result of this intermediate step - df_PCA_done.csv )

In the Clustering part ( ML Project Pairs Trading - K-means.ipynb ,ML Project Pairs Trading - DBSCAN.ipynb, ML Project Pairs Trading - Agglomerative clustering.ipynb  ) K-means, DBSCAN ,  Agglomerative clustering were used. Han suggests, based on empirical results, K=500, alpha=10 and alpha=30 to be used as parameters for clustering. Results are:

        						k-means	k=500		DBSCAN	α=10		Agglomerative	α=30
        						paper	my		paper	my		paper	my

	Number of clusters				129	218		2	1.54		188	220
	Number of stocks in total			3157	3286		3157	3286		3157	3286
	Number of stocks in clusters			1257	1990		376	429		769	798
	Number of outliers				1900	1296		2781	2847		2388	2488
	Number of stocks in the biggest cluster		63	67		336	420		107	102
	Number of stocks in the second biggest cluster	51	56		32	19		47	42
	Number of stocks in the third biggest cluster	45	50		6	9		29	24

Table shows that results are partially matched. Best matches are for Agglomerative clustering. ( results of this intermediate step - df_KMeans.csv,df_DBSCAN.csv,df_Agglomerative.csv )

In the Trading part ( ML Project Pairs Trading - Trading.ipynb ) I implemented trading, based on cluster analyses and returns from previous month, creating pairs of companies with the biggest difference in returns ( if that difference in returns is bigger than cross section std ).

				Kmeans L	Kmeans L my	Kmeans S	Kmeans S my	Kmeans L-S	Kmeans L-S my	S&P500	Reversal

        Mean_return		0.272		0.230625	0.056		-0.084778	0.215		0.081431	0.126	0.239
	Standard_deviation	0.206		0.212544	0.165		0.17495		0.092		0.042076	0.151	0.243
	Sharpe_ratio		1.318		1.085069	0.339		-0.484585	2.336		1.935302	0.835	0.983
	Downside_deviation	0.158		0.218144	0.133		0.158453	0.046		0.033855	0.112	0.141
	Sortino_ratio		1.719		1.057212	0.423		-0.535036	4.713		2.40525		1.123	1.693
	Maximum_drawdown	-0.505		-0.529242	-0.58		-0.976305	-0.113		-0.049242	-0.502	-0.503
	Calmar_ratio		0.537		0.43576474	0.097		0.08683557	1.9		-1.65368994	0.251	0.475


				DBSCAN L	DBSCAN L my	DBSCAN S	DBSCAN S my	DBSCAN L-S	DBSCAN L-S my	S&P500	Reversal

	Mean_return		0.263		0.208677	0.007		-0.068496	0.256		0.073559	0.126	0.239
	Standard_deviation	0.18		0.172489	0.149		0.139599	0.126		0.050642	0.151	0.243
	Sharpe_ratio		1.457		1.209803	0.044		-0.490661	2.039		1.452528	0.835	0.983
	Downside_deviation	0.131		0.185531	0.127		0.123229	0.063		0.047867	0.112	0.141
	Sortino_ratio		2.007		1.12476		0.052		-0.555842	4.065		1.536731	1.123	1.693
	Maximum_drawdown	-0.481		-0.515399	-0.796		-0.9561		-0.149		-0.073734	-0.502	-0.503
	Calmar_ratio,0.537	0.547		-0.40488437	0.008		0.07164104	1.717		-0.9976266	0.251	0.475


				Agglo L		Agglo L my	Agglo S		Agglo S my	Agglo L-S	Agglo L-S my	S&P500	Reversal

	Mean_return		0.277		0.205389	0.029		-0.069772	0.248		0.075189	0.126	0.239
	Standard_deviation	0.187		0.189144	0.157		0.16136		0.092		0.038853	0.151	0.243
	Sharpe_ratio		1.481		1.085888	0.183		-0.432398	2.692		1.935197	0.835	0.983
	Downside_deviation	0.139		0.199562	0.134		0.139828	0.048		0.034162	0.112	0.141
	Sortino_ratio		1.998		1.029199	0.215		-0.498983	5.15		2.20094		1.123	1.693
	Maximum_drawdown	-0.424		-0.496052	-0.595		-0.953343	-0.123		-0.055043	-0.502	-0.503
	Calmar_ratio,0.537	0.653		-0.41404732	0.048		0.07318667	2.013		-1.36600476	0.251	0.475


These tables show a significant difference between paper and my results. These are results without a transaction fee.
( results of this intermediate step - KMeans_trading_results.csv,DBSCAN_trading_results.csv,Agglomerative_trading_results.csv )

Further development of this topic:
-  stationarity check
- optimal cluster number 
- optimal PCA components number - a lot of PCA components could be not optimal for clustering ( Bellman (1966), Berkhin ( 2006 ))
- minimal number of company characteristics - Green ( GHZ2017 ) found that after 2003 just 2 (chempia , nincr ) parameters are statistically significant in explaining returns.Hwang and Rubesam (2020) found 2 more additional  ( ear, herf )  
- lag of company characteristics - since data are mostly quarterly published can we use older data if there is no new ones?
- combaning outlier detection with all 3 clustering algorithms
- determine distribution of returns in function of previous month return difference ( closer or far away from std )



