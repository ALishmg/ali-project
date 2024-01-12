import numpy as np
from sklearn.decomposition import FastICA


np.random.seed(0)
n_samples = 200
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)

A = np.array([[1, 1], [0.5, 2]])
X = np.dot(S, A.T)  # Mixed signals

ica = FastICA(n_components=2)
S_ica = ica.fit_transform(X)

print("Estimated sources after ICA:\n", S_ica)