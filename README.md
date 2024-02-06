# Pnet
Implementation of P-Net as a flexible deep learning tool to generate insights from genetic features.

Current pytorch implementation in revision.

## Model
Pnet uses the Reactome hierarchical graph as underlying structure to reduce the number of connections in a fully connected feed forward neural network. The sparse layers connect only known pathways. This limits the number of parameters to be learnt in a meaningful way and facilitate learning via gradient descent and leads to more generalizable models. 

## Installation
Once you cloned the repo, cd into it and run """pip install -e . """
