# GraphSAGE_waveShrink&GraphSAGE_waveThresh

An extension of GraphSage. Use  Wave_Thresh&Waveshrink method.

We take the repository of [graphsage-simple](https://github.com/williamleif/graphsage-simple) for reference.

# Usage

## Running examples

Execute `python -m graphsage.model` to run the example. It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this. 

## Code Structure

    citeseer/: Example datasets of citeseer .
    
    cora/: Example datasets of cora.
    
    pubmed-data/: Example datasets of pubmed.
    
    graphsage/: graphsage library.
      aggregators.py: Set of modules for aggregating embeddings of neighbors.
      encoders.py:Encodes a node's using 'convolutional' GraphSage approach
      models.py: models of graphsage & Dataset parsers & Runs experiments.
    
