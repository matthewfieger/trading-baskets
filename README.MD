
Balanced Baskets: A New Approach to Trading and Hedging Risks
============================================================
David H. Bailey and Marcos Lopez de Prado, Journal of Investment Strategies (Risk Journals), Vol.1(4), Fall 2012

Abstract
--------
A basket is a set of instruments that are held together because its statistical profile delivers a desired goal, such as hedging or trading, which cannot be achieved through the individual constituents or even subsets of them. Multiple procedures have been proposed to compute hedging and trading baskets, among which balanced baskets have attracted significant attention in recent years. Unlike Principal Component Analysis (PCA) style of methods, balanced baskets spread risk or exposure across their constituents without requiring a change of basis. Practitioners typically prefer balanced baskets because their output can be understood in the same terms for which they have developed an intuition.

We review three methodologies for determining balanced baskets, analyze the features of their respective solutions and provide Python code for their calculation. We also introduce a new method for reducing the dimension of a covariance matrix, called Covariance Clustering, which addresses the problem of numerical ill-conditioning without requiring a change of basis.


ERC Results
-----------
	##### SOLUTION 1 #####
	##### Holdings #####
	[[ 1.        ]
	 [-4.11046593]
	 [ 3.27414209]]
	##### CtR #####
	[0.33334582237125071, 0.33333279003696176, 0.33332138759178387]
	##### CtB #####
	[0.13570164382941163, 0.05125111093630208, 0.079026034284033222]
	##### Stats #####
	RMSD=9.9828516336e-06
	#Iter=233
	##### SOLUTION 2 #####
	##### Holdings #####
	[[ 1.        ]
	 [-4.11057203]
	 [ 3.27427439]]
	##### CtR #####
	[0.33332125901785298, 0.33333302860185965, 0.33334571238029487]
	##### CtB #####
	[0.13569463952917465, 0.051250956063429755, 0.079030352275060833]
	##### Stats #####
	RMSD=9.98536861723e-06
	#Iter=544
	##### SOLUTION 3 #####
	##### Holdings #####
	[[ 1.        ]
	 [-4.1105712 ]
	 [ 3.27427336]]
	##### CtR #####
	[0.33332144816158094, 0.33333303337591697, 0.33334551846250465]
	##### CtB #####
	[0.13569469317943314, 0.051250958285671513, 0.079030317604099981]
	##### Stats #####
	RMSD=9.82894797621e-06
	#Iter=709

Source
------
Bailey, David H. and Lopez de Prado, Marcos, Balanced Baskets: A New Approach to Trading and Hedging Risks (May 24, 2012). Journal of Investment Strategies (Risk Journals), Vol.1(4), Fall 2012. Available at SSRN: http://ssrn.com/abstract=2066170 or http://dx.doi.org/10.2139/ssrn.2066170