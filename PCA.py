#!/usr/bin/env python
# Covariance clustering algo
# On 20120516 by MLdP <lopezdeprado@lbl.gov>
import numpy as np
from itertools import combinations
from scipy import delete

#-------------------------------------------
#-------------------------------------------
class ClusterCov:
	def __init__(self,cov,minSize):
		# Class constructor
		self.cov=cov
		self.minSize=minSize
		self.cn=[]
		self.clusters=[]
#-------------------------------------------
	def solve(self,clusterSize=2):
		# Determine the optimal clustering of the covariance matrix
		size=self.cov.shape[0]
		bundles=[]
		while self.cov.shape[0]>self.minSize:
			cn=[]
			comb=self.comb(range(self.cov.shape[0]),clusterSize)
			for trial in comb:
				# Copy matrix
				cov=self.cov[:]
				# Bundle
				cov=self.bundle(cov,trial)
				# Compute condition number
				evalues=np.linalg.eigvalsh(cov)
				cn_=max(evalues)/min(evalues)
				cn.append(cn_)
			# Determine optimal cluster
			sol=self.findSolution(cn)
			if sol<=0:break
			# Store solution
			self.cn.append(sol)
			bundles.append(comb[cn.index(self.cn[-1])])
			self.cov=self.bundle(self.cov[:],bundles[-1])
		self.clusters=self.getClusters(bundles,size)
		return
#-------------------------------------------
	def bundle(self,cov,indices):
		# Bundle the covariance matrix
		diag,a=0,0
		# Add clustering column and row
		for i in indices:
			a+=cov[:,i]
		# Compute diagonal element
		for i in indices:
			diag+=a[i]
		cov=self.expandCov(cov,a,diag)
		# Remove clustered columns and rows
		for i in range(len(indices)):
			cov=delete(cov,indices[i]-i,0)
			cov=delete(cov,indices[i]-i,1)
		return cov
#-------------------------------------------
	def expandCov(self,cov,array,diag):
		# Expand the covariance matrix
		b=array[:]
		b.shape=(b.shape[0],1)
		cov=np.hstack((cov,b))
		b=np.append(array,diag)
		b.shape=(1,b.shape[0])
		cov=np.vstack((cov,b))
		return cov
#-------------------------------------------
	def comb(self,iterable,groupSize):
		# Generate all combinations
		comb=[item for item in combinations(iterable,groupSize)]
		return comb
#-------------------------------------------
	def findSolution(self,cn):
		# Find the optimal solution among candidates
		sol=max(cn)
		for i in cn:
			if i<sol and i>0:sol=i
		return sol
#-------------------------------------------
	def getClusters(self,bundles,size):
		# Determine clusters' constituents
		clusters=range(size)
		for i in bundles:
			a=[]
			for j in i:
				a.append(clusters[j])
			clusters.append(a)
			for j in range(len(i)):
				clusters.pop(i[j]-j)
		return clusters
#-------------------------------------------
#-------------------------------------------
def main():
	# Example using the ClusterCov class
	path='Covariance.csv'
	#0) Parameters
	minSize=2
	#1) Load the covariance matrix
	cov=np.genfromtxt(path,delimiter=',') # numpy array #2) Cluster
	cluster=ClusterCov(cov,minSize)
	cluster.solve()
	3#) Report results
	print cluster.cov.shape
	print cluster.cov
	print cluster.cn
	print cluster.clusters
	return
#-------------------------------------------
# Boilerplate
if __name__=='__main__': main()