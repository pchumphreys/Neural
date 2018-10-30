I am working on some code to implement time-dependent neural dynamics using tensorflow. The code currently includes forward connections, as well as recurrence within a layer. However, feedback connections are not yet included.

The cleanest code is for a rate-based implementation, although I am also testing a spike based version.

So far the code focuses on reproducing the behaviour in [Learning by the dendritic prediction of somatic spiking](https://www.ncbi.nlm.nih.gov/pubmed/24507189)
