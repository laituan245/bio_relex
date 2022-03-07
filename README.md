# Joint Biomedical Entity and Relation Extraction with Knowledge-Enhanced Collective Inference

##  Instructions
The code has been tested with Python 3. To install the dependencies, please run:
```
pip install -r requirements.txt
```

We use two datasets in this work:
+ ADE. We conducted 10-fold cross-validation. The dataset can be downloaded from [here](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/).
+ BioRelEx. The train and dev sets can be downloaded [here](https://github.com/YerevaNN/BioRelEx/releases).
The test set is unreleased and can only be evaluated using [CodeLab](https://competitions.codalab.org/competitions/20468).

After downloading the datasets, please create a new folder `resources` and put the datasets into that folder.
Overall, the folder structure of the entire repo should look like:
```
...
models/
pymetamap/
resources/
--- ade/
------- ade_full.json
------- ade_split_0_test.json
------- ade_split_0_train.json
....
------- ade_split_9_test.json
------- ade_split_9_train.json
------- ade_types.json
--- biorelex/
------- train.json
------- dev.json
--- umls_embs.pkl
--- umls_rels.txt
--- umls_reltypes.txt
--- umls_semtypes.txt
--- text2graph.pkl
scorer/
.gitignore
ade_train.sh
...
```
Additional files in the `resources` folder include:
+ The files `umls_rels.txt`, `umls_reltypes.txt`, and `umls_semtypes.txt` can be extracted directly from UMLS (to use UMLS, you need to request [access permission](https://www.nlm.nih.gov/research/umls/index.html)).
+ `umls_embs.pkl` contains the embeddings of [Maldonado et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6568073/) and also the embeddings of the UMLS definition sentences. Note that some UMLS concepts may not have any definition sentence.
+ `text2graph.pkl` is a cache that maps each text input in the datasets into a graph structure of all the concepts and relations from UMLS that can be potentially relevant (found by [MetaMap](https://metamap.nlm.nih.gov/)).

For training, please refer to the scripts `ade_trainer.py` and `trainer.py`. For example, to train a basic model for BioRelEx, you can simply run:
```
python trainer.py
```

**Note**: If you want me to send you UMLS-related files, please email me at tuanml2@illinois.edu (together with some proof that you have access to UMLS). I am not putting UMLS-related files online because of the UMLS licensing issue.


There are some redundant code in this repo. I am going to remove them soon.
