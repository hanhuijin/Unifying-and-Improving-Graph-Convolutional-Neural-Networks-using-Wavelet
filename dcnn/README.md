# DCNN_WaveThresh&DCNN_WaveShrink
An extension of DCNN. Use  Wave_Thresh&Waveshrink method.

We take the repository of [DCNN](https://github.com/jcatw/dcnn) for reference.(An implementation of [Diffusion-Convolutional Neural Networks](http://papers.nips.cc/paper/6212-diffusion-convolutional-neural-networks.pdf) [1] in Theano and Lasagne.)

## Usage
## Running examples

Execute`python -m client.run `to run the example. 



## Code Structure
    client/: Client code for running from the command line.
      parser.py: Parses command line args into configuration parameters.
      run.py: Runs experiments.
    
    data/: Example datasets.
    
    python/: DCNN library.
      data.py: Dataset parsers.
      layers.py: Lasagne internals for DCNN layers.
      models.py: User-facing end-to-end models that provide a scikit-learn-like interface.
      params.py: Simple container for configuration parameters.
      util.py: Misc utility functions.



