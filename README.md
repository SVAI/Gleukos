# Gleukos

*adapted from [NCBI-Hackathons/GoodDoc](https://github.com/NCBI-Hackathons/GoodDoc) with some tweaks for analysis-driven projects*

*instructions in italics can be deleted as sections are filled in*

*most fields are optional, Conclusion and Important Resources are required*

## Please cite our work -- here is the ICMJE Standard Citation:

### ...and a link to the DOI: *You can make a free DOI with zenodo, synapse, figshare, or other resources <link>*


## Abstract *: Summarize everything in a few sentences.* 
We use data science approach by leveraging the drug screening data to reveal enriched pathways in treating plexiform neurofibromas. Our resuts suggested novel pathways for this disease and some drug combinations for the treatment.

## Introduction *: What's the problem? Why should we solve it?*
Plexiform neurofibromas (pNFs) is a serious type of Neurofibromatosis. It impact ~40% of people with Neurofibromatosis Type 1 (NF1). Other than the other types of Neurofibromatosis, this subtype can lead to death. In this work, we use a data science approach to suggest some drug combinations for this disease.

## Methods *: How did we go about solving it?*
Based on the drug screening data provided, we first linked it to a more general CHEMBL dataset of drug targets. Then many relevant targets are revealed for the compounds which showed at 1 micromolar potency to the pNFs cell lines. We further analyzed the associated pathways of the relavent targets and obtained 10 enriched ones. From the enriched pathways, we can select drug combinations that are more robust in treating pNFs.

## Results *: What did we observe? Figures are great!*
We observed 10 enriched pathways as shown below. Some are very interesting because in addition to some cancer-related pathways, we found pathways like proteosome, calcium signaling, thyroid hormone. Those provide us with novel targets to consider for drug discovery.

(https://github.com/SVAI/Gleukos/blob/master/output_18_1.png)

## Conclusion/Discussion: 
By integrating the drug screening data and multiple datasets, we found several pathways associated to the plexiform neurofibromas. We also provided some drug combinations suggestion for the biological assays next step.

### Please make sure you address ALL of the following:

#### *1. What additional data would you like to have*
None. All are uploaded on this repo.

#### *2. What are the next rational steps?* 
The next steps would do drug safety check and do animal model experiments.

#### *3. What additional tools or pipelines will be needed for those steps?*
Drug development pipelines like assays and animal model.

#### *4. What skills would additional collaborators ideally have?*
Experimental skills to test our prediction.

## Reproduction: *How to reproduce the findings!*
You can reproduce all the findings by running https://github.com/SVAI/Gleukos/blob/master/nf.ipynb



