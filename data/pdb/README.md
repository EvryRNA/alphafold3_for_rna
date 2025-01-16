# PDB data

This folder contains the raw `.pdb` data for the benchmark. 

You can download the data at this [link](https://drive.google.com/file/d/1xGc9eBM4SYE4UnMI04n6EOa82lfVIjL7/view?usp=sharing). 

## Folder

It is composed of the following datasets:

- `CASP_RNA`
- `RNA3DB` (corresponds to `RNA3DB_0`)
- `RNA3DB_LONG`
- `RNA_PUZZLES`
- `RNASOLO`

Note that for each folder, you have the `NATIVE`, `ALIGNED` and `PREDS` subfolders. 

For the `RNA3DB_0` dataset, the predictions with or without context can be find with the names of the `.pdb` files: structures with names `alphafold3c_` correspond 
to the structures predicted from the context and then extracted. 
