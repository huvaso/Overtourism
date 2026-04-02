# From Flows to Graphs: Data-Driven Insights on Latent Overtourism with Frequent Pattern Mining

## Created by
**Author:** Hugo Alatrista-Salas <br />
**Co-authors** - Gaël Chareyron, Sonia Djebali, Imen Ouled-Dlala and Nicolas Travers  <br />
**Maintainer:** Hugo Alatrista-Salas <br />
**Contact Details:** hugo.alatrista_salas@devinci.fr <br />
**Institution:** De Vinci Higher Education, De Vinci Research Center, Paris, France 
<br />


**Note:** use Python 3.9.*

## Description

Overtourism presents complex and often hidden challenges for urban environments, impacting residents, infrastructure, and visitor satisfaction. This repository contains the code and resources associated with our research on detecting **latent overtourism**—the early, subtle warning signs of excessive tourism before visible disruptions occur.

Our approach introduces a **data-driven methodology** combining graph modeling and pattern mining techniques to uncover hidden tourist movement dynamics.

---

## Methodology

The proposed framework is based on the following key steps:

1. **Data Collection**  
  User-generated content from Tripadvisor is leveraged to reconstruct tourist activity.

2. **Graph Modeling**  
   A **temporal circulation multidigraph** is built to represent tourist mobility across locations.

3. **Frequent Subgraph Mining**  
   Recurring movement patterns are extracted using frequent pattern mining algorithms.

4. **Spatio-Temporal Analysis**  
   Identified patterns are analyzed across space and time to detect:
   - Emerging hotspots  
   - High-pressure urban zones  

5. **Attractiveness Modeling**  
   A **Huff-based probabilistic model** is used to evaluate dynamic attractiveness of locations.

**Note:** The full paper is available at https://link.springer.com/chapter/10.1007/978-3-032-05727-3_28

## Citation

If you use this code or find it helpful in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-05727-3_28,
author="Alatrista-Salas, Hugo
and Chareyron, Ga{\"e}l
and Djebali, Sonia
and Ouled-Dlala, Imen
and Travers, Nicolas",
editor="Chrysanthis, Panos K.
and N{\o}rv{\aa}g, Kjetil
and Stefanidis, Kostas
and Zhang, Zheying
and Quintarelli, Elisa
and Zumpano, Ester",
title="From Flows to Graphs: Data-Driven Insights on Latent Overtourism with Frequent Pattern Mining",
booktitle="New Trends in Database and Information Systems",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="327--342",
isbn="978-3-032-05727-3"
}
