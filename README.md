# ATCS-project
The final project for the ATCS course

The `SCAN_dataset.csv` contains analogies from 2 domains: science and metaphor. There is a possibility for an analogy to also include 
`alternatives`, which essentially are synonyms for the target word. Still, most analogies in this dataset do not have any alternatives.

The `SCAN_examples.txt` file contains detailed examples for items from the `SCAN_dataset.csv`. There are 25 detailed examples 
for each analogy type present in the original dataset. The examples are phrased in a neutral manner, with average word length.
The examples will be extended with 
- active vs passive
- simple wording VS complex wording

## Running the inference

```
python run.py
```

## Notebook

The notebook downloads the files from Git and triggers the inference for the default params and only 3 data points - at least for now, 
since I thought we can use it as a way to verify a working end-to-end implementation of the inference pipeline.