# Computational-Neuroscience_IISER'19
This repository contains collection of mathematical models of neuroplasticity. I worked at IISER Pune in summer of 2019 to explore a new found field of interest for me. It turned out much better than I anticipated The focus is mainly on understanding the mechanisms, or specifically, the biological mechanisms that stabilize the activity dependent Hebbian forms of learning. One of the postdocs (Dr. Mahajan) from the Computational Neurobiology lab of IISER Pune had worked on this before. He focused on the role of ER stores in mediating metaplasticity (change in synaptic plasticity) at dendritic spines. The research focused on the involvement of IP3 receptors as calcium releasing channel for the store calcium. It turns out that the ER has one more channel for the release of calcium. It's a receptor which can be activated (or inactivated) by adding an appropriate amount of a diterpenoid called ryanodine - hence its name 'ryandine receptor' (RyR). Now before you get too excited about using ryanodine and trying out some experiments on yourselves, I should warn you that ryanodine is actually poisonous. You don't want malfunctioning RyRs in your cells, because that may lead to irregular muscle contractions, arrythmia, seizures and maybe death. But anyway, my role was to explore the contribution of RyRs in mediating a spine specific form of metaplasticity. In this repository, I've mainly shared the Python 3 codes for modelling a Hodgkin - Huxley neuron (cos my supervisor thought that it'll be a good starting point into Computational Neuroscience), the model of NMDA receptor (which is a voltage and calcium dependent channel, allowing external calcium into the cell) - because it is the one that contributes strongly to metaplasticity, a model of RyR and finally a unified spine containing various channels and ER stores. I'm using part of Dr. Mahajan's code and adding my RyR model to it and seeing its physiological validity as of now.
