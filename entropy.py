from math import log2

#  calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i] * log2(q[i]) for i in range(len(p))])

# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate entropy H(P)
def entropy(p):
	return -sum([p[i] * log2(p[i]) for i in range(len(p))])

p = [0.10, 0.1,0.1]
q = [0.2, 0.2,0.2]

print(kl_divergence(p,q))