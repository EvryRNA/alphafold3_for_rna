# AlphaFold 3 for RNA benchmark


## Installation

To get all the data (the `.pdb` structures predicted), you can find them on this [link](https://drive.google.com/file/d/1OR7Gol0hjB-CfyR9DpzCa8mHq81miW5Q/view?usp=sharing). 

You can then unzip in the `data` folder. 

To install the required library for the visualisation, you can use:

```bash
make install_rna_assessment
```

## Compute metrics

If you want to reproduce the metrics computation, we used the [RNAdvisor](https://github.com/EvryRNA/rnadvisor) tool, using:

```bash
make compute_metrics
```

Otherwise, the scores are already computed and available in the `data/output` folder.

## Statistics

To get the statistics of the datasets, you can use the following command:

```bash
make stats
```

To compute the different types of interactions, you can use:

```bash
make count_interactions
```


## Visualisations

To reproduce the visualisation available in `data/plots`, you can use:

```bash
make viz_alphafold
```

It will save the plots in the `data/plots` folder.


## Authors

- [Clément Bernard](https://github.com/clementbernardd)
- Guillaume Postic
- Sahar Ghannay
- Fariza Tahi