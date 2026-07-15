# 🧬 TCM-TopoGraph: Geometric and Topological Representations of the Human Meridian System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TCM-TopoGraph** is an Explainable AI (XAI) framework that utilizes Geometric Deep Learning (GDL) and Persistent Homology to model the complex, non-Euclidean physiological network of the human meridian system[cite: 3, 4]. By transforming noisy physiological and pharmacological data into robust topological representations, this framework bridges computational topology with clinical praxis[cite: 3].

---

## 🌟 Live Demo & Deployment
This project utilizes a dual-deployment architecture for optimal accessibility:
* **Interactive Landing Page (GitHub Pages):** [Insert your GitHub Pages URL here]
* **Live AI Dashboard (Streamlit Cloud):** [Insert your Streamlit Cloud URL here]

---

## 📊 Dataset Characteristics
The framework is built upon a real-world, multi-modal heterogeneous graph[cite: 1, 3]:
* **Anatomical Nodes (361 Acupoints):** Features multi-modal embeddings including spatial coordinates and physiological priors[cite: 2, 3].
* **Pharmacological Nodes (714 Herbs):** Contains biochemical indices and native medicinal tropism data[cite: 2, 3].
* **Directed Flow Edges:** Represents sequential physiological signal propagation along the 14 primary meridians.
* **Heterogeneous Tropism Edges:** Bipartite mappings connecting external pharmacological herbs to specific anatomical target networks.

---

## ⚙️ Core Analytical Modules
The execution engine (`app.py`) provides four primary interactive modules[cite: 4]:

1. **In-silico Discovery (Topo-GNN):** Calculates Geometric Tropism Rankings and extracts "Novel Targets" using PyVis heterogeneous pathways[cite: 4].
2. **3D Manifold Mapping:** Projects physiological pathways onto a curved 3D WebGL space, proving the utilization of Geodesic distances over classical Euclidean metrics[cite: 4].
3. **TDA Filtration Simulator:** Interactively tracks real-time changes in Betti numbers (β₀ for disconnected components, β₁ for topological cycles) across varying distance radiuses[cite: 4].
4. **Real-Time Analytics:** Tracks live topological biomarkers, simulating homeostatic regulation pre- and post-clinical intervention[cite: 4].

---

## 📂 Repository Structure

```text
TCM-TopoGraph/
│
├── index.html                     # Static landing page for GitHub Pages
├── app.py                         # Streamlit application engine / GUI
├── requirements.txt               # Python dependencies
├── README.md                      # Repository documentation
│
├── TCM_HeteroGraph_Builder.ipynb  # Google Colab / Jupyter pipeline for graph building
│
├── TCM14_Nodes_Base64.csv         # Core Dataset: Acupoint node data
├── Acupoints_With_Images.csv      # Core Dataset: Topological coordinates
└── ViThuoc_final.csv              # Core Dataset: Herb & Tropism mappings

```

🚀 Local Installation & Execution
To run the interactive framework on your local machine:

1. Clone the repository:
```bash
git clone [https://github.com/vothikimanh1007/TCM-TopoGraph.git](https://github.com/vothikimanh1007/TCM-TopoGraph.git)
cd TCM-TopoGraph

```

2. Install dependencies:
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt

```

3. Launch the Streamlit application:
```bash
streamlit run app.py

```

4. Google Colab Pipeline (Optional):
To rebuild the graph tensors from scratch, upload the three core .csv files to a Google Colab instance and run TCM_HeteroGraph_Builder.ipynb

📖 Citation
If you use this framework or dataset in your research, please cite our paper currently under peer review:
```snipset
@article{ ,
  title={Unified Geometric and Topological Representations of the Human Meridian System: A Deep Learning Framework Leveraging Symmetry and Persistence},
  author={ },
  journal={ },
  year={2026},
  note={}
}

```

License: This project is licensed under the MIT License. See the LICENSE file for details.
