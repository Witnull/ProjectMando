# MANDO-HGT REBUILD

[![python](https://img.shields.io/badge/python-3.10.12-blue)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/Ubuntu-22.04-orange)](https://releases.ubuntu.com/jammy/)
[![Static Badge](https://img.shields.io/badge/Docker-latest-teal)](https://docker.com/)
[![Static Badge](https://img.shields.io/badge/CUDA-12.1-green)]

Service: `mando-hgt-rebuild`

## Directories tree

Some folders may not exist on repo, this is expected.

```
.
|
+---- assets 
|
+---- data 
|
+---- experiments (VERY IMPORTANT)
|        |
|        |
|        +---- ge-sc-data (DATASET), OVERWRITE THIS from Old MANDO |
|        |
|        +---- models (Auto Generated)
|        +---- logs (Auto Generated) -- main result are here
|        |
|        +---- node_classification.py (VERY IMPORTANT) | DO NOT OVERWRITE THIS -- main running
|        |
|        +---- graph_classification.py (VERY IMPORTANT) | DO NOT OVERWRITE THIS -- main running
|
|
|
|
|
+---- forensics 
|
+---- Mando (Contains the Old Mando, may not need to put it in, this is for ease of access to the old script to test it)
|
+---- ge-sc-data (Contains the Dataset repo below) | Extract from zip maybe?
|        |
|        +---- ijicai20
|        +---- node_classification
|        +---- others
|        \
|
|
+---- models (Auto Generated)
|
|
|
+---- newMethods (Our apply to tét with new datasets)
|      |
|      |
|      +---- archive : Dataset - SoliAudit 
|      +---- FromCSVToFile.py - Extract the source_code from csv to .sol files
|      +---- result_parse.py - Parsing the result from ./newMethods/logs/test_logs/<id>/ folder 
|              +---- output might like this file: ./test_sampleDataset_1000.txt
|
+---- process_graphs (VERY IMPORTANT) | DO NOT OVERWRITE THIS
|
+---- sco_models (VERY IMPORTANT) | DO NOT OVERWRITE THIS
|
+---- utils (Self added stuff)
|
+---- logs (Auto Generated)
|
|
|
+---- cm.sh interesting command to use the graph_classifier.py and node_classifier.py, missing some gpickle files to run tho.
|
+---- How2Run.md
|
+---- graph_classifier.py ( Maybe custom train)
|
+---- node_classifier.py ( Maybe custom train)
|
+---- note.txt (Drawbacks or interesting stuff in paper)
|
+---- visualize.py (For visualize dataset for now)
|
|
\



```

## Required dependencies

### Repositories (recommended using zip download):

Follow above Dir tree to install and setup

#### Old MANDO:

`https://github.com/MANDO-Project/ge-sc.git`
\
\*\* copy the experiments folder's data.

#### Dataset:

`https://github.com/minhnn-tiny/ge-sc-dataset.git`
\
\*\* Dataset for test 

---

## Build docker

To using docker (RECOMMENDED)

`docker compose up` ( --build : to rebuild the docker with latest changes)

`docker exec -ti <container name/id> /bin/bash`

More information refer to the Docker documentation:
`https://docs.docker.com/compose/`

---

## Running

#### Note: X - May not need to run / ignore

### Processing data to graphs phase

Please read the displayed Info when running the script.

1. CFG: `python process_graphs/control_flow_graph_generator.py`

2. CG: `python process_graphs/call_graph_generator.py`

3. CG_CFG: `python process_graphs/combination_call_graph_and_control_flow_graph_helper.py`

4. compressed GPICKLES: `python process_graphs/byte_code_control_flow_graph_generator.py`
   - This is for the baseline models, please follow the number order

---

### Training the baselines models phase

#### Graph Classification (Coarse-grained - contract level)

Training Phase

```bash
python -m experiments.graph_classification --epochs 50 --repeat 20
```

To show the result table

```bash
python -m experiments.graph_classification --result
```

---

#### Node Classification (Fine-grained - line level)

Training Phase

```bash
python -m experiments.node_classification --epochs 50 --repeat 20
```

To show the result table

```bash
python -m experiments.node_classification --result
```


# References:

## SoliAudit
```
@INPROCEEDINGS{8939256,
  author={Liao, Jian-Wei and Tsai, Tsung-Ta and He, Chia-Kang and Tien, Chin-Wei},
  booktitle={2019 Sixth International Conference on Internet of Things: Systems, Management and Security (IOTSMS)}, 
  title={SoliAudit: Smart Contract Vulnerability Assessment Based on Machine Learning and Fuzz Testing}, 
  year={2019},
  volume={},
  number={},
  pages={458-465},
  keywords={Smart contracts;Blockchain;Machine learning;Feature extraction;Security;Fuzzing;Smart contract;vulnerability;fuzz testing;machine learning},
  doi={10.1109/IOTSMS48152.2019.8939256}}

```
---

<!--
# ------ IGNORE BELOW ------

---

# Methods

| Category                          | Methods                          | CMD |
| --------------------------------- | -------------------------------- | --- |
| Original Heterogeneous GNN        | metapath2vec                     |     |
| Original Homogeneous GNNs         | LINE <br> node2vec               |     |
| The best Buggy F1 scores of MANDO | Node features of the best scores |     |

---

### Test baselines models phase

---

### Training HGT models phase

| Category                                  | Methods                  | CMD |
| ----------------------------------------- | ------------------------ | --- |
| MANDO-HGT with Node Features Generated by | NodeType One Hot Vectors |     |
|                                           | metapath2vec             |     |
|                                           | LINE                     |     |
|                                           | node2vec                 |     |

---

### Test HGT models phase

---

### Visualize phase

6. process_graphs/dgl_graph_generator.py X
7. process_graphs/solidifi_reader.py X

8. node_classifier.py
9. node_classifier.py
10. visualize.py

11. sco_models/dataloader.py
12. sco_models/graph_utils.py
13. sco_models/model_hgt.py
14. sco_models/opcodes.py
15. sco_models/utils.py
16. sco_models/visualization.py

Flag: `y0uf0undtheFl4g_m4nd0h67` -->
