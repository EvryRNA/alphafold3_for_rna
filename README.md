<div align="center">
<a href="https://doi.org/10.1093/nargab/lqae048" target="_blank" title="Go to article"><img width="500px" alt="logo" src="data/plots/img/graphical_abstract.png"></a>
<a name="readme-top"></a>

# AlphaFold 3 for RNAs benchmark

This repository is the source code for the visualisations of [AlphaFold 3 benchmark for RNA](https://evryrna.ibisc.univ-evry.fr/evryrna/alphafold3)
<br> It also contains the aligned structures for each prediction for the <br> five test sets: `RNA_PUZZLES`, `CASP_RNA`, `RNASOLO`, `RNA3DB_0` and `RNA3DB_LONG`.

[![Article][article_img]][article_url]
[![License][repo_license_img]][repo_license_url]


<a href="https://www.biorxiv.org/content/10.1101/2024.06.13.598780v2" target="_blank" title="Go to article"><img width="400px" alt="logo" src="src/assets/img/video3d.gif"></a>
<a name="readme-top"></a>



</div>

![](data/plots/img/alphafold3_best_worst.png)


# AlphaFold 3 for RNAs benchmark

This repository is the source code for the visualisations of the article named: `Has AlphaFold 3 reached its success for RNAs?`

![AlphaFold 3 for RNA](data/plots/img/graphical_abstract.png)


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

## Citation


If you use this code, please cite the following paper:

```
Has AlphaFold 3 reached its success for RNAs?
Clement Bernard, Guillaume Postic, Sahar Ghannay, Fariza Tahi
bioRxiv 2024.06.13.598780; 
doi: https://doi.org/10.1101/2024.06.13.598780
```

Or using the bibtex format:

```bibtex
@article{Bernard2024,
  author = {Clement Bernard and Guillaume Postic and Sahar Ghannay and Fariza Tahi},
  title = {Has AlphaFold 3 reached its success for RNAs?},
  year = {2024},
  journal = {bioRxiv},
  doi = {10.1101/2024.06.13.598780},
  note = {Preprint},
  url = {https://doi.org/10.1101/2024.06.13.598780}
}
```

## Authors

- [Clément Bernard](https://github.com/clementbernardd)
- Guillaume Postic
- Sahar Ghannay
- Fariza Tahi

<!-- Links -->

[article_img]: https://img.shields.io/badge/BioRxiv-Article-blue?style=for-the-badge&logo=none
[article_url]: https://www.biorxiv.org/content/10.1101/2024.06.13.598780v2
[repo_license_img]: https://img.shields.io/badge/license-Apache_2.0-red?style=for-the-badge&logo=none
[repo_license_url]: https://github.com/EvryRNA/alphafold3_for_rna/blob/main

