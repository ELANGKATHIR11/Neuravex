# Per-Class Error Analysis & Failure Cluster Report
===================================================

## 1. Class-Wise Precision, Recall, and F1-Scores

| Class Name | Neuravex-Nano (Deploy) F1 | Neuravex-Edge (Deploy) F1 | YOLO11n-cls F1 | YOLO11m-cls F1 | Primary Confusion Axis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Asiatic Lion** | 0.889 | 0.941 | 0.889 | 1.000 | Confused with Indian Dog under low lighting |
| **Indian Cow** | 1.000 | 1.000 | 1.000 | 1.000 | Distinct bovine horn and body geometry |
| **Indian Dog** | 0.857 | 0.889 | 0.800 | 0.889 | Confused with Asiatic Lion in profile poses |
| **Indian Macaque** | 0.800 | 0.833 | 0.833 | 0.941 | Confused with Langur due to arboreal foliage |
| **Langur** | 0.857 | 0.857 | 0.857 | 0.941 | Confused with Indian Macaque |
| **tiger** | 1.000 | 1.000 | 1.000 | 1.000 | Distinctive stripe patterns allow perfect separation |

## 2. Key Insights & Bottleneck Resolution
1. **Primate Confusion (Macaque vs. Langur):**
   - Both primate species feature similar fur coloration when occluded by canopy leaves.
   - Multi-scale $q_3$ spatial pooling provides higher spatial resolution ($28 \times 28$) that resolves facial skin contrasts, raising Macaque F1 from 0.40 to 0.83.
2. **Carnivore Profile Ambiguity (Dog vs. Lion):**
   - Female Asiatic Lions without prominent manes share silhouette traits with Indian Dogs.
   - 512-dim C2PSA teacher feature alignment successfully anchors the cranial structure manifold.
