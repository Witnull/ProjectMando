# MANDO REBUILD

[![python](https://img.shields.io/badge/python-3.10.12-blue)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/Ubuntu-22.04-orange)](https://releases.ubuntu.com/jammy/)
[![Static Badge](https://img.shields.io/badge/Docker-latest-teal)](https://docker.com/)


Service: `mando-hgt-rebuild`

## Required dependencies

Repository:
`https://github.com/MANDO-Project/ge-sc.git`
\
\*\* Some of the data needed to build and run in this repository.

---

## Build docker

To using docker
`docker compose up`
`docker exec -ti <container name/id> /bin/bash`
Or 
`docker build -t <image name/id> .`
`docker run -itv .:/app/mando-hgt <image name/id>`


More information refer to the Docker documentation: 
`https://docs.docker.com/compose/`

---

## Running

#### Note: X - May not need to run

### Processing data to graphs

1. process_graphs/call_graph_generator.py
2. process_graphs/control_flow_graph_generator.py
3. process_graphs/combination_call_graph_and_control_flow_graph_helper.py
4. process_graphs/byte_code_control_flow_graph_generator.py

---

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


Flag: `y0uf0undtheFl4g_m4nd0h67`