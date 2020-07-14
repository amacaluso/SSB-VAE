# SSB-VAE
Self-Supervised Bernoulli Autoencoders for Semi-Supervised Hashing


This repository contains the code to reproduce the results presented in the paper 
*Self-Supervised Bernoulli Autoencoders for Semi-Supervised Hashing*.

we investigate the robustness of hashing methods based on variational autoencoders 
to the lack of supervision, focusing on two semi-supervised approaches currently in use. 
The first augments the training objective of the variational autoencoder to jointly model 
the distribution over the data and the class labels. The second approach exploits the 
annotations to define an additional *pairwise* loss that enforces a consistency 
between the similarity in the code (Hamming) space and the similarity in the label space. 
Our experiments on text and image retrieval tasks show that, as expected, both methods 
can significantly increase the quality of the hash codes. The pairwise approach can exhibit 
an advantage when the number of labelled points is large. However, we found that this method 
can degrade quickly and loose its advantage when the amount of labelled samples decreases. 
To circumvent this problem, we propose a novel supervision method in which the model uses 
its own predictions of the label distribution to implement the pairwise objective. We found 
that, compared to the best baseline, this procedure yields similar performance in 
fully-supervised settings but yields significantly better results when labelled data is scarce.