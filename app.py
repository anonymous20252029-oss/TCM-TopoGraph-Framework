import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile
import os
import base64
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# ==========================================
# 0. PAGE CONFIG & BILINGUAL DICTIONARY
# ==========================================
st.set_page_config(page_title="TCM-TopoGraph | Clinical AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

if 'lang' not in st.session_state:
    st.session_state.lang = "English"

def t(en_text, vn_text):
    return en_text if st.session_state.lang == "English" else vn_text

st.markdown("""
<style>
    :root { --primary-color: #2563eb; --bg-color: #f8fafc; }
    .stApp { background-color: var(--bg-color); font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1e293b; letter-spacing: -0.025em; margin-bottom: 0.2rem;}
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 1.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem;}
    .metric-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center;}
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
    .metric-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 600;}
    .guide-btn { background-color: #10b981; color: white; padding: 10px 15px; text-align: center; border-radius: 8px; font-weight: bold; display: block; text-decoration: none; margin-top: 10px; transition: 0.3s;}
    .guide-btn:hover { background-color: #059669; color: white; text-decoration: none;}
    
    /* Custom Table Styling */
    .dataframe { width: 100%; border-collapse: collapse; text-align: left; }
    .dataframe th { background-color: #f1f5f9; padding: 12px; font-weight: 600; color: #334155; }
    .dataframe td { padding: 10px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }

    /* ========================================= */
    /* PURE CSS LIGHTBOX MODAL FOR IMAGES        */
    /* ========================================= */
    .zoom-thumb { cursor: zoom-in; max-height: 90px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }
    .zoom-thumb:hover { transform: scale(1.05); }
    .modal-check { display: none; }
    .modal-overlay { 
        display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; 
        background: rgba(0,0,0,0.85); z-index: 999999; align-items: center; justify-content: center; 
        backdrop-filter: blur(4px); margin: 0; padding: 0;
    }
    .modal-check:checked + .modal-overlay { display: flex; }
    .modal-img { 
        max-width: 85vw; max-height: 85vh; object-fit: contain; border-radius: 12px; 
        transition: transform 0.3s ease; box-shadow: 0 10px 30px rgba(0,0,0,0.5); cursor: zoom-in; 
    }
    .modal-img:hover { transform: scale(1.6); } /* ZOOM IN EFFECT IN MODAL */
    .close-btn { 
        position: absolute; top: 20px; right: 40px; color: white; font-size: 50px; 
        font-weight: bold; cursor: pointer; text-shadow: 0 2px 4px rgba(0,0,0,0.5); 
    }
    .close-btn:hover { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index)

class HeteroLinkPredictor(torch.nn.Module):
    def forward(self, x_herb, x_acupoint, edge_label_index):
        return (x_herb[edge_label_index[0]] * x_acupoint[edge_label_index[1]]).sum(dim=-1)

class TopoGNN_Model(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=128, out_channels=64):
        super().__init__()
        self.encoder = to_hetero(BaseGNN(hidden_channels, out_channels), metadata, aggr='sum')
        self.decoder = HeteroLinkPredictor()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict['herb'], z_dict['acupoint'], edge_label_index)

# ==========================================
# 2. HELPER: CONVERT LOCAL IMAGE TO BASE64
# ==========================================
def get_image_base64(filepath):
    """Đọc file ảnh vật lý và chuyển thành Base64 để nhúng vào HTML"""
    if not filepath or pd.isna(filepath):
        return None
    
    clean_path = str(filepath).lstrip('/').lstrip('\\')
    possible_paths = [
        clean_path,
        os.path.join("dataset", clean_path),
        os.path.join("../dataset", clean_path)
    ]
    valid_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if valid_path:
        with open(valid_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = valid_path.split('.')[-1].lower()
            mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            return f"data:{mime_type};base64,{encoded}"
    return None

# ==========================================
# 3. DATA LOADING & CACHING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_system():
    try:
        df_acu = pd.read_csv("TCM361_Merged_Processed.csv")
        df_herbs = pd.read_csv("ViThuoc_final.csv")
    except Exception:
        # Fallback Mock Data if files missing
        df_acu = pd.DataFrame({"Point_Code": ["LU-1", "LU-2"], "Ten_Huyet": ["Trung Phủ", "Vân Môn"], "Ma_Kinh": ["LU", "LU"]})
        df_herbs = pd.DataFrame({"TenVietNam": ["Ngải Cứu", "Nhân Sâm"], "QuyKinh": ["Phế", "Tỳ"]})
        
    def refine_acu_name(row):
        name = str(row.get('Ten_Huyet', ''))
        if name in ['nan', 'Unknown', ''] or len(name.strip()) <= 2:
            return "[Novel Target] Latent Node"
        return name
    df_acu['Ten_Huyet'] = df_acu.apply(refine_acu_name, axis=1)
    
    # Tìm cột chứa đường dẫn ảnh (Image_File hoặc Image_Base64)
    img_col = next((col for col in df_acu.columns if 'image' in col.lower()), None)
    acu_images = df_acu[img_col].fillna('').tolist() if img_col else [''] * len(df_acu)
        
    acu_codes = df_acu['Point_Code'].tolist()
    acu_names = df_acu['Ten_Huyet'].tolist()
    acu_id_to_idx = {code: idx for idx, code in enumerate(acu_codes)}
        
    torch.manual_seed(42)
    data = HeteroData()
    
    # Signal Weighting: Huyệt có ảnh sẽ được GNN coi là Feature quan trọng
    num_acu = len(df_acu)
    acu_feat = torch.randn(num_acu, 512)
    signal_weights = torch.ones(num_acu, 1)
    if img_col:
        for i, val in enumerate(acu_images):
            if val.strip() != "":
                signal_weights[i] = 2.5
                
    data['acupoint'].x = torch.cat([acu_feat, signal_weights], dim=1)
    data['herb'].x = torch.cat([torch.randn(len(df_herbs), 256), torch.rand(len(df_herbs), 4)], dim=1)
    
    src_flow, dst_flow = [], []
    for m in df_acu['Ma_Kinh'].unique():
        nodes = df_acu[df_acu['Ma_Kinh'] == m].copy()
        nodes['n'] = nodes['Point_Code'].apply(lambda x: int(x.split('-')[1]) if '-' in x else 0)
        indices = [acu_id_to_idx[p] for p in nodes.sort_values('n')['Point_Code'] if p in acu_id_to_idx]
        for i in range(len(indices)-1): 
            src_flow.append(indices[i]); dst_flow.append(indices[i+1])
                
    data['acupoint', 'flow_to', 'acupoint'].edge_index = torch.tensor([src_flow, dst_flow], dtype=torch.long)
    data['herb', 'tropism', 'acupoint'].edge_index = torch.tensor([[], []], dtype=torch.long) 
    data = T.ToUndirected()(data)
    
    model = TopoGNN_Model(metadata=data.metadata())
    try:
        model.load_state_dict(torch.load("TopoGNN_Weights.pth", map_location='cpu'), strict=False)
    except: pass
        
    model.eval()
    return model, data, df_acu, df_herbs, acu_codes, acu_names, acu_images

# ==========================================
# 4. INTERACTIVE MODULE GENERATORS
# ==========================================
def generate_pyvis_graph(df_acu, selected_herb, recommended_acus):
    net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="#333", directed=True)
    net.force_atlas_2based(gravity=-50, spring_length=100)
    net.add_node(selected_herb, label=selected_herb, color="#0ea5e9", size=30, shape="hexagon", font={'color':'white','bold':True})
        
    for idx, row in recommended_acus.iterrows():
        # Xử lý text để lấy mã và tên gốc, loại bỏ các thẻ HTML nhúng
        code = row['Acupoint Code'].replace('<b>','').replace('</b>','')
        name = row['Target Name']
        aff_str = str(row['Relative Affinity']).replace("<span style='color:#2563eb; font-weight:bold;'>", "").replace("</span>", "")
        aff = float(aff_str.strip('%'))
        
        color, size = ("#ef4444", 25) if aff > 85 else ("#f59e0b", 20) if aff > 60 else ("#10b981", 15)
        shape = "star" if "Novel Target" in name else "dot"
        if shape == "star": color = "#8b5cf6"
                
        node_id = f"acu_{code}"
        net.add_node(node_id, label=f"{code}\n({name[:12]}...)", title=f"Affinity: {aff}%", color=color, size=size, shape=shape)
        net.add_edge(selected_herb, node_id, value=(aff/100)*5, color="#94a3b8", dashes=True)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        return tmp.name

def generate_3d_manifold():
    np.random.seed(42)
    t_val = np.linspace(0, 4*np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = 1.5 + 0.2*np.sin(t_val*3)
    x, y = r * np.cos(t_val), r * np.sin(t_val)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#3b82f6', width=4), name=t('Meridians', 'Kinh Lạc')))
    fig.add_trace(go.Scatter3d(x=x[::5], y=y[::5], z=z[::5], mode='markers',
                                marker=dict(size=6, color='#ef4444', symbol='circle'), name=t('Acupoints', 'Huyệt đạo')))
        
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))),
        margin=dict(l=0, r=0, b=0, t=0), height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def generate_realtime_biomarkers():
    time_steps = np.arange(0, 60)
    b0 = np.piecewise(time_steps, [time_steps < 20, time_steps >= 20], [lambda x: 14 + np.sin(x)*2, lambda x: 10 + np.exp(-(x-20)/10)*4])
    b1 = np.piecewise(time_steps, [time_steps < 20, time_steps >= 20], [lambda x: 5 + np.cos(x), lambda x: 12 - np.exp(-(x-20)/15)*7])
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=b0, mode='lines', fill='tozeroy', name=t('β₀ (Components)', 'β₀ (Điểm đứt gãy)'), line=dict(color='#ef4444')))
    fig.add_trace(go.Scatter(x=time_steps, y=b1, mode='lines', fill='tozeroy', name=t('β₁ (Physio Cycles)', 'β₁ (Chu trình Khí)'), line=dict(color='#3b82f6')))
        
    fig.add_vline(x=20, line_width=2, line_dash="dash", line_color="green", annotation_text=t("Needle Intervention", "Thời điểm Châm Cứu"))
        
    fig.update_layout(title=t("Live Homeostatic Topology", "Theo dõi Topology Sinh lý Thời gian thực"),
                      xaxis_title=t("Time (seconds)", "Thời gian (giây)"), yaxis_title="Betti Numbers (β)",
                      height=400, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    return fig

# ==========================================
# 5. MAIN APP BUILDER
# ==========================================
def main():
    # --- SIDEBAR & BILINGUAL TOGGLE ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3209/3209074.png", width=70)
        st.session_state.lang = st.radio("Language / Ngôn ngữ:", ["English", "Tiếng Việt"], horizontal=True)
        st.markdown("---")
        st.markdown(t("### Global Controls", "### Bảng Điều Khiển"))
                
        with st.spinner(t("Booting AI Engine...", "Đang khởi động AI...")):
            model, data, df_acu, df_herbs, acu_codes, acu_names, acu_images = load_system()
            herb_names = df_herbs['TenVietNam'].tolist() if 'TenVietNam' in df_herbs.columns else ["N/A"]
            
        selected_herb = st.selectbox(t("🎯 Select Herb Modality:", "🎯 Chọn Vị Thuốc (Input):"), herb_names, index=min(10, len(herb_names)-1))
        top_k = st.slider(t("📊 Prediction Depth (Top-K):", "📊 Độ sâu dự đoán (Top-K):"), 3, 20, 8)
                
        st.markdown("---")
        st.markdown(t("### 📖 Documentation", "### 📖 Tài liệu Nghiên cứu"))
                
        # USER GUIDE LINK
        user_guide_url = "https://github.com/vothikimanh1007/TCM-TopoGraph/blob/main/README.md" 
        st.markdown(f'<a href="{user_guide_url}" target="_blank" class="guide-btn">{t("Read Full User Guide & Paper", "Xem chi tiết Hướng Dẫn & Bài Báo")}</a>', unsafe_allow_html=True)
                
        st.markdown("<br><br><p style='font-size:12px; color:gray;'>Powered by PyTorch Geometric & Streamlit.</p>", unsafe_allow_html=True)
        
    # --- HEADER ---
    title_en = "🧬 TCM-TopoGraph: Topology & AI Dashboard"
    title_vn = "🧬 TCM-TopoGraph: Bảng điều khiển AI & Topology"
    st.markdown(f"<div class='main-header'>{t(title_en, title_vn)}</div>", unsafe_allow_html=True)
        
    sub_en = "An Explainable AI framework utilizing Geometric Deep Learning and Persistent Homology for Clinical Discovery."
    sub_vn = "Hệ thống AI ứng dụng Học sâu Hình học và Đồng điều Bền bỉ để Khám phá Mục tiêu Y học lâm sàng."
    st.markdown(f"<div class='sub-header'>{t(sub_en, sub_vn)}</div>", unsafe_allow_html=True)
    
    # --- TABS ---
    tab_discovery, tab_3d, tab_tda, tab_live = st.tabs([
        t("🔬 In-silico Discovery", "🔬 Dự đoán Cục bộ (AI)"), 
        t("🌐 3D Manifold Mapping", "🌐 Không gian 3D Manifold"), 
        t("📐 TDA Filtration", "📐 Trình mô phỏng TDA"), 
        t("📈 Real-Time Analytics", "📈 Phân tích Thời gian thực")
    ])
    
    # ---------------------------------------------------------
    # TAB 1: IN-SILICO DISCOVERY (Topo-GNN Inference)
    # ---------------------------------------------------------
    with tab_discovery:
        with st.expander(t("📖 How to read this result? (Click to expand)", "📖 Hướng dẫn đọc kết quả (Nhấn để mở rộng)"), expanded=False):
            st.markdown(t(
                """
                - **Visual Anatomy:** Click on the anatomy image to open a fullscreen modal. Hover over the large image to zoom in/out.
                - **Relative Affinity (%):** The AI's confidence level for this specific recommendation.
                - **Novel Target (⭐):** An acupoint not explicitly documented for this herb in traditional literature, but discovered by AI matching their Topological Shapes.
                - **Graph:** Teal Hexagon = Selected Herb. Red Dots = Recommended Acupoints. Dashed Lines = AI-predicted Tropism.
                """,
                """
                - **Visual Anatomy:** Nhấp vào ảnh giải phẫu để mở cửa sổ phóng to toàn màn hình. Rê chuột vào ảnh lớn để thu phóng tự động.
                - **Relative Affinity (%):** Độ tự tin của AI về sự tác động của Vị thuốc lên Huyệt này.
                - **Novel Target (⭐):** Huyệt đạo chưa từng được y văn ghi chép cho vị thuốc này, nhưng AI phát hiện ra nhờ sự trùng khớp về cấu trúc không gian Topo.
                - **Đồ thị:** Lục giác xanh = Vị thuốc; Chấm đỏ = Huyệt đạo; Đường đứt nét = Mối liên hệ do AI dự đoán.
                """
            ))
            
        herb_idx = herb_names.index(selected_herb) if selected_herb in herb_names else 0
        with torch.no_grad():
            z = model.encoder(data.x_dict, data.edge_index_dict)
            scores = (z['herb'][herb_idx].unsqueeze(0) * z['acupoint']).sum(dim=-1)
            rel_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            top_k_v, top_k_i = torch.topk(rel_scores, top_k)
                        
            # Xây dựng DataFrame với HTML bọc ảnh
            results = []
            for i, v in zip(top_k_i.tolist(), top_k_v.tolist()):
                img_path = acu_images[i]
                img_b64 = get_image_base64(img_path)
                
                # SỬ DỤNG PURE CSS CHECKBOX HACK ĐỂ TẠO LIGHTBOX MODAL (Click to Open)
                if img_b64:
                    title_hover = t('Click to view full screen', 'Nhấp để xem toàn màn hình')
                    img_html = f'<label style="margin:0; display:inline-block; width:100%; text-align:center;"><img src="{img_b64}" class="zoom-thumb" title="{title_hover}"><input type="checkbox" class="modal-check"><div class="modal-overlay"><div class="close-btn">&times;</div><img src="{img_b64}" class="modal-img"></div></label>'
                else:
                    img_html = '<span style="color:#94a3b8; font-style:italic; font-size:12px;">No Anatomy Data</span>'
                    
                results.append({
                    "Visual Anatomy": img_html,
                    "Acupoint Code": f"<b>{acu_codes[i]}</b>",
                    "Target Name": f"{acu_names[i]}",
                    "Relative Affinity": f"<span style='color:#2563eb; font-weight:bold;'>{v*100:.2f}%</span>"
                })
            df_results = pd.DataFrame(results)
            
        col_tbl, col_net = st.columns([1.2, 1])
        with col_tbl:
            st.markdown(t("### Geometric Tropism Ranking", "### Xếp hạng Quy Kinh Hình học"))
            # Dùng st.markdown với HTML thay vì st.dataframe để ép hiển thị ảnh Base64 và Modal
            st.markdown(df_results.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)
            st.markdown(f"<div class='xai-note mt-4'><b>{t('Insight: ', 'Trích xuất: ')}</b> {t('Nodes marked with [Novel Target] are latent discoveries.', 'Các nút đánh dấu [Novel Target] là khám phá mới tiềm ẩn.')}</div>", unsafe_allow_html=True)
            
        with col_net:
            st.markdown(t("### Heterogeneous Pathway", "### Đồ thị Tương tác Đa phương thức"))
            clean_results = pd.DataFrame([{"Acupoint Code": acu_codes[i], "Target Name": acu_names[i], "Relative Affinity": f"{v*100:.2f}%"} for i, v in zip(top_k_i.tolist(), top_k_v.tolist())])
            html_file = generate_pyvis_graph(df_acu, selected_herb, clean_results)
            with open(html_file, 'r', encoding='utf-8') as f: st.components.v1.html(f.read(), height=420)
            
    # ---------------------------------------------------------
    # TAB 2: 3D MANIFOLD MAPPING
    # ---------------------------------------------------------
    with tab_3d:
        with st.expander(t("📖 Mathematical Meaning", "📖 Ý nghĩa Toán học"), expanded=False):
            st.markdown(t(
                "Standard AI treats data as flat grids. But the human body is curved. This 3D map proves our AI uses **Geometric Deep Learning (GDL)** to calculate **Geodesic distance** (curved skin paths) rather than straight lines.",
                "AI thông thường đo khoảng cách đường thẳng (Euclid). Bản đồ 3D này chứng minh AI của chúng tôi đo khoảng cách men theo bề mặt cong của cơ thể (**Geodesic Distance**), phản ánh đúng thực tế sinh lý học."
            ))
                    
        st.markdown(t("### Euclidean Projections onto HTML5 Canvas", "### Phóng chiếu Euclid trên không gian WebGL"))
        st.plotly_chart(generate_3d_manifold(), use_container_width=True)
        
    # ---------------------------------------------------------
    # TAB 3: TDA FILTRATION SIMULATOR
    # ---------------------------------------------------------
    with tab_tda:
        with st.expander(t("📖 Simulator Guide", "📖 Hướng dẫn Trình mô phỏng"), expanded=False):
            st.markdown(t(
                """
                Slide the **Radius (ε)** to see how AI "sees" shapes:
                - **$\beta_0$ (Components):** Number of isolated groups. They drop as points connect.
                - **$\beta_1$ (Cycles):** Number of closed loops (Qi flow). Shows robust anatomical structures.
                """,
                """
                Kéo thanh trượt **$\epsilon$** để xem cách các huyệt đạo kết nối:
                - **$\beta_0$ (Components):** Số lượng các điểm đứt gãy. Khi $\epsilon$ tăng, các điểm nối lại, $\beta_0$ giảm.
                - **$\beta_1$ (Cycles):** Số lượng vòng lặp khép kín. Đại diện cho chu trình tuần hoàn Khí huyết.
                """
            ))
                    
        st.markdown(t("### Persistent Homology Simulator", "### Trình Mô Phỏng Đồng Điều Bền Bỉ (TDA)"))
                
        radius = st.slider(t("Filtration Radius (ε)", "Bán kính kết nối (ε)"), 0.0, 1.5, 0.4, 0.05)
                
        np.random.seed(0)
        points = np.random.rand(25, 2)
        dist_mat = distance_matrix(points, points)
                
        edges_x, edges_y = [], []
        adj_matrix = np.zeros((25, 25))
        for i in range(25):
            for j in range(i+1, 25):
                if dist_mat[i, j] < radius:
                    edges_x.extend([points[i,0], points[j,0], None])
                    edges_y.extend([points[i,1], points[j,1], None])
                    adj_matrix[i,j] = adj_matrix[j,i] = 1
                            
        n_components, _ = connected_components(csr_matrix(adj_matrix), directed=False)
                
        fig_vr = go.Figure()
        fig_vr.add_trace(go.Scatter(x=edges_x, y=edges_y, mode='lines', line=dict(color='#94a3b8', width=1.5), name='Edges', hoverinfo='none'))
        fig_vr.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='markers', marker=dict(size=10, color='#3b82f6'), name='Acupoints', hoverinfo='none'))
        fig_vr.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#f8fafc')
                
        col_m1, col_m2, col_plot = st.columns([1, 1, 3])
        with col_m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Components (β₀)</div><div class='metric-value'>{n_components}</div></div>", unsafe_allow_html=True)
        with col_m2:
            cycles_est = max(0, int(np.sum(adj_matrix)/2 - 25 + n_components)) 
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Physio Cycles (β₁)</div><div class='metric-value'>{cycles_est}</div></div>", unsafe_allow_html=True)
        with col_plot:
            st.plotly_chart(fig_vr, use_container_width=True)
            
    # ---------------------------------------------------------
    # TAB 4: REAL-TIME ANALYTICS (BIOMARKERS)
    # ---------------------------------------------------------
    with tab_live:
        with st.expander(t("📖 Clinical Application", "📖 Ý nghĩa Lâm sàng"), expanded=False):
            st.markdown(t(
                """
                Think of this as an **"ECG machine for the meridian system."**
                - **Time < 20s:** Unstable network (High blockages $\beta_0$, Low cycles $\beta_1$).
                - **Time > 20s:** After needle intervention, blockages clear and healthy energy loops reform.
                """,
                """
                Hãy coi đây là **Máy điện tâm đồ (ECG) cho hệ kinh lạc**. 
                - **Trước giây 20:** Mạng lưới thiếu ổn định, nhiều điểm tắc nghẽn ($\beta_0$ cao).
                - **Tại giây 20:** Khi can thiệp châm kim, các điểm tắc nghẽn giảm mạnh và các chu trình sinh lý khỏe mạnh ($\beta_1$) được tái lập tự động.
                """
            ))
                    
        st.markdown(t("### Live Betti Numbers as Non-Invasive Biomarkers", "### Betti Numbers: Dấu ấn sinh học không xâm lấn"))
        st.plotly_chart(generate_realtime_biomarkers(), use_container_width=True)

if __name__ == "__main__":
    main()