Mean Flows for DNA Sequence Applications 

This is the mean flows application for DNA sequence design, see original paper here (https://arxiv.org/pdf/2402.05841).

Please install the packages in requirements.txt, (lightning)

Toy simplex experiment can be performed by running toy_simplex.py. 
See lightning_logs/version_[v_num] to see training metrics. 


TODO: 
Your model is showing KL divergence minimization during training! 
We still have a few issues though, the actual training is flattening with the huber loss and blows up with mse, figure out some balance
Also, find a way to save a model after a thorough training run and then test it on the 512 K samples that the paper does. 