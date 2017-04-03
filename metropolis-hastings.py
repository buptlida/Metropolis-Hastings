#!usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from pylab import *
import random

class MH():
    # def __int__(self, p, q, samples):
    #     self.samples = samples #采样个数
    #     self.chain = np.zeros(samples) #构造的马尔科夫链
    #     self.p = p #待采样概率密度函数
    #     self.q = q #建议概率密度函数

    def __init__(self, p, q, samples):
        self.samples = samples # integer number of samples to do, typically > 5,000
        self.chain = np.zeros(samples) # initialize list of samples to 0
        self.p = p # posterior distribution
        self.q = q # proposal distribution

    def alpha(self, candidate, current):
        return min(1, self.p(candidate) * self.q(current) / (self.p(current) * self.q(candidate)))

    def generate_candidate(self, i, mu, sigma):
        return self.chain[i] + random.normalvariate(mu, sigma)

    def sample(self, mu, sigma, burn_in = 250):
        self.chain[0] = random.normalvariate(mu, sigma)  # initial value
        u = np.random.uniform(0, 1, self.samples)
        for i in range(1, self.samples):
            candidate = self.generate_candidate(i-1, mu, sigma)
            if u[i] < self.alpha(candidate, self.chain[i-1]):
                self.chain[i] = candidate
            else:
                self.chain[i] = self.chain[i-1]
        self.chain = self.chain[burn_in : self.samples]

    def plot_result(self):
        figure(1)
        count = 0
        for i in range(1,self.chain.size):
            if self.chain[i] == self.chain[i-1]:
                count +=1
        print count
        hist(self.chain, bins = 50)
        ylabel('Frequency')
        xlabel('Value')
        title('Histogram of Samples')

        figure(2)
        plot(self.chain)
        ylabel('Values')
        xlabel('Interation')
        title('Trace Markov Values')
        show()

if __name__=='__main__':
    def PosteriorDistribution(x):
        mu1 = 3
        mu2 = 10
        v1 = 10
        v2 = 3
        return 0.3*exp(-(x-mu1)**2/v1) + 0.7*exp(-(x-mu2)**2/v2)

    def ProposalDistribution(x):
        return exp(-(x-5)**2/100)

    model = MH(PosteriorDistribution, ProposalDistribution, 1000000)
    model.sample(5, 10)
    model.plot_result()