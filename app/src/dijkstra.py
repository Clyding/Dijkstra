# app.py
import streamlit as st
import json
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components
import datetime

st.set_page_config(page_title="Baltimore Transit — Geo Graph + Dijkstra + Critical Points", layout="wide")
st.title("Baltimore Inner Harbor — Geospatial Graph Model")

# --------------------------
# Human-readable names
# --------------------------
NODE_NAMES = {
    "P": "Penn Station (Transit Hub)",
    "A": "National Aquarium",
    "F": "Fort McHenry",
    "B": "M&T Bank Stadium (Oriole Park)",
    "C": "Convention Center (Light Rail Stop)",
    "D": "Fells Point",
    "E": "Federal Hill",
    "G": "Harbor East",
    "H": "Harbor Point",
    "I": "Inner Harbor (Harborplace)",
    "J": "Jonestown",
    "K": "Locust Point / Ferry Area",
    "L": "Little Italy",
    "M": "Mount Vernon",
    "O": "Oriole Park / Camden Yards (alt)",
    "Q": "Lexington Market / Downtown",
    "R": "Ridgely's Delight",
    "S": "Otterbein",
    "T": "Inner Harbor Water Taxi Dock"
}

# --------------------------
# Geospatial coordinates (node letter -> (lat, lon))
# --------------------------
NODE_COORDS = {
    "P": (39.3079, -76.6157),   # Penn Station
    "A": (39.2857, -76.6081),   # National Aquarium
    "F": (39.2813, -76.5803),   # Fort McHenry
    "B": (39.2839, -76.6217),   # M&T Bank Stadium
    "C": (39.2851, -76.6187),   # Convention Center
    "D": (39.2830, -76.6050),   # Fells Point (approx)
    "E": (39.2789, -76.6120),   # Federal Hill
    "G": (39.2833, -76.6026),   # Harbor East
    "H": (39.2784, -76.6000),   # Harbor Point
    "I": (39.2850, -76.6132),   # Inner Harbor
    "J": (39.2913, -76.6052),   # Jonestown
    "K": (39.2708, -76.5930),   # Locust Point / Ferry Area
    "L": (39.2867, -76.6030),   # Little Italy
    "M": (39.3009, -76.6158),   # Mount Vernon
    "O": (39.2821, -76.6188),   # Oriole Park / Otterbein
    "Q": (39.2911, -76.6208),   # Lexington Market
    "R": (39.2847, -76.6220),   # Ridgely's Delight
    "S": (39.2930, -76.6140),   # Downtown Core / Otterbein approx
    "T": (39.2844, -76.6105)    # Water Taxi Dock
}

# --------------------------
# Default adjacency (undirected)
# weights are minutes (includes wait estimates for transit)
# --------------------------
DEFAULT_ADJ = {
    "P": [("M", 6, "walk"), ("L", 8, "walk"), ("C", 12, "walk"), ("S", 10, "walk")],
    "M": [("P", 6, "walk"), ("B", 9, "light_rail"), ("C", 7, "light_rail")],
    "L": [("P", 8, "walk"), ("Q", 10, "walk"), ("D", 6, "walk")],
    "Q": [("L", 10, "walk"), ("C", 7, "walk")],
    "C": [("A", 3, "walk"), ("T", 2, "walk"), ("B", 10, "walk"), ("D", 9, "walk"),
          ("E", 7, "walk"), ("G", 8, "walk"), ("I", 5, "walk"), ("R", 5, "walk"), ("S", 8, "walk"), ("P", 12, "walk")],
    "A": [("C", 3, "walk")],
    "T": [("C", 2, "walk"), ("F", 20, "water_taxi")],
    "F": [("T", 20, "water_taxi")],
    "B": [("M", 9, "light_rail"), ("C", 10, "walk"), ("J", 4, "walk"), ("K", 6, "walk")],
    "J": [("B", 4, "walk"), ("K", 6, "walk")],
    "K": [("B", 6, "walk"), ("J", 6, "walk"), ("I", 12, "walk"), ("E", 9, "walk"), ("C", 14, "walk")],
    "D": [("C", 9, "walk"), ("L", 6, "walk"), ("R", 6, "walk")],
    "G": [("L", 5, "walk"), ("I", 6, "walk"), ("C", 8, "walk")],
    "I": [("G", 6, "walk"), ("C", 5, "walk"), ("K", 12, "walk")],
    "E": [("C", 7, "walk"), ("K", 9, "walk")],
    "R": [("C", 5, "walk"), ("D", 6, "walk")],
    "S": [("P", 10, "walk"), ("C", 8, "walk")]
}

# --------------------------
# Utilities: normalize undirected adjacency (mirror edges & dedupe)
# --------------------------
def normalize_undirected(adj: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    g: Dict[str, List[Tuple[str, float, str]]] = {}
    for u, nbrs in adj.items():
        g.setdefault(u, [])
        for entry in nbrs:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                v = entry[0]
                w = float(entry[1])
                mode = entry[2] if len(entry) >= 3 else "walk"
                g[u].append((v, w, mode))
                g.setdefault(v, [])
                g[v].append((u, w, mode))
            else:
                raise ValueError("Adj list entries must be (node, weight [, mode])")
    # deduplicate identical triples
    for u in list(g.keys()):
        seen = set()
        dedup: List[Tuple[str, float, str]] = []
        for v,w,mode in g[u]:
            key = (v, float(w), mode)
            if key not in seen:
                dedup.append((v,w,mode))
                seen.add(key)
        g[u] = dedup
    return g

# --------------------------
# Load adjacency (sidebar option)
# --------------------------
st.sidebar.header("Graph Data")
use_default = st.sidebar.checkbox("Use built-in graph (recommended)", value=True)
if use_default:
    ADJ = normalize_undirected(DEFAULT_ADJ)
else:
    adj_text = st.sidebar.text_area("Paste adjacency JSON (node -> [[nbr,weight,mode],...])",
                                    value=json.dumps(DEFAULT_ADJ, indent=2), height=250)
    try:
        ADJ = normalize_undirected(json.loads(adj_text))
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        ADJ = normalize_undirected(DEFAULT_ADJ)

NODES = sorted(list(ADJ.keys()))

# --------------------------
# Graph algorithms: matrix, dijkstra, dfs AP/bridges
# --------------------------
def adjacency_matrix(adj: Dict[str, List[Tuple]]) -> pd.DataFrame:
    nodes = sorted(adj.keys())
    n = len(nodes)
    mat = np.full((n, n), np.inf)
    idx = {node:i for i,node in enumerate(nodes)}
    for u in nodes:
        mat[idx[u], idx[u]] = 0.0
        for v,w,mode in adj[u]:
            mat[idx[u], idx[v]] = w
    df = pd.DataFrame(mat, index=nodes, columns=nodes)
    return df

def dijkstra(adj: Dict[str, List[Tuple]], source: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    dist = {node: float("inf") for node in adj}
    parent = {node: None for node in adj}
    dist[source] = 0.0
    pq: List[Tuple[float,str]] = []
    heappush(pq, (0.0, source))
    while pq:
        d,u = heappop(pq)
        if d > dist[u]:
            continue
        for v,w,mode in adj[u]:
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heappush(pq, (nd, v))
    return dist, parent

def reconstruct_path(parent: Dict[str,str], target: str) -> List[str]:
    if target not in parent:
        return []
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def find_bridges_and_articulation_points(adj: Dict[str, List[Tuple]]):
    time = 0
    disc: Dict[str,int] = {}
    low: Dict[str,int] = {}
    parent: Dict[str, str] = {}
    ap_set: Set[str] = set()
    bridges: List[Tuple[str,str]] = []

    for u in adj:
        disc[u] = -1
        low[u] = -1
        parent[u] = None

    def dfs(u: str):
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        children = 0
        for v,w,mode in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] is None and children > 1:
                    ap_set.add(u)
                if parent[u] is not None and low[v] >= disc[u]:
                    ap_set.add(u)
                if low[v] > disc[u]:
                    bridges.append((u,v))
            elif parent[u] != v:
                low[u] = min(low[u], disc[v])

    for node in adj.keys():
        if disc[node] == -1:
            dfs(node)

    bridges_norm = [tuple(sorted((u,v))) for u,v in bridges]
    bridges_norm = sorted(list(set(bridges_norm)))
    return sorted(list(ap_set)), bridges_norm

# --------------------------
# UI Controls (sidebar)
# --------------------------
st.sidebar.header("Route & Simulation Controls")
src = st.sidebar.selectbox("Source", NODES, index=NODES.index("P") if "P" in NODES else 0)
tgt = st.sidebar.selectbox("Target", NODES, index=NODES.index("A") if "A" in NODES else 1)
compute_button = st.sidebar.button("Compute shortest path & analyze")

failure_mode = st.sidebar.selectbox("Simulate failure:", ["None", "Remove Edge", "Remove Node"])
fail_edge = None
fail_node = None
if failure_mode == "Remove Edge":
    edges_flat = []
    seen = set()
    for u in sorted(ADJ.keys()):
        for v,w,mode in ADJ[u]:
            a,b = sorted((u,v))
            if (a,b) not in seen:
                seen.add((a,b))
                edges_flat.append(f"{a} - {b} ({w}m, {mode})")
    chosen = st.sidebar.selectbox("Edge to remove", edges_flat)
    uu = chosen.split()[0]
    vv = chosen.split()[2]
    fail_edge = (uu, vv)
elif failure_mode == "Remove Node":
    fail_node = st.sidebar.selectbox("Node to remove", NODES)

show_ap = st.sidebar.checkbox("Show articulation points on map", value=True)
show_bridges = st.sidebar.checkbox("Show bridges on map", value=True)
show_all_edges = st.sidebar.checkbox("Show all edges (polylines)", value=True)

# --------------------------
# simulate removal helpers
# --------------------------
def remove_edge(adj: Dict[str, List[Tuple]], u: str, v: str):
    newadj = {k:[(x,y,z) for x,y,z in lst] for k,lst in adj.items() for k in [k]}
    for a,b in [(u,v),(v,u)]:
        if a in newadj:
            newadj[a] = [t for t in newadj[a] if t[0] != b]
    return newadj

def remove_node(adj: Dict[str, List[Tuple]], node: str):
    newadj = {}
    for u, lst in adj.items():
        if u == node:
            continue
        newadj[u] = [(v,w,mode) for v,w,mode in lst if v != node]
    return newadj

# baseline APs/bridges
base_ap, base_bridges = find_bridges_and_articulation_points(ADJ)

# --------------------------
# Map builder (folium)
# --------------------------
MODE_COLOR = {"walk": "gray", "light_rail": "green", "water_taxi": "blue", "unknown": "black"}

def build_map(adj: Dict[str, List[Tuple]], highlight_path: List[str]=None, ap_list: List[str]=None, bridges: List[Tuple[str,str]]=None):
    center = [39.2904, -76.6122]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    # scale line thickness by weight
    all_weights = [w for u in adj for _,w,_ in adj[u]] or [1]
    min_w, max_w = min(all_weights), max(all_weights)
    def stroke_width(w):
        if max_w == min_w:
            return 3
        return 2 + 6 * ((w - min_w) / (max_w - min_w))

    # draw edges (unique undirected)
    seen = set()
    for u in adj:
        if u not in NODE_COORDS:
            continue
        for v,w,mode in adj[u]:
            a,b = sorted((u,v))
            if (a,b) in seen:
                continue
            seen.add((a,b))
            if v not in NODE_COORDS:
                continue
            color = MODE_COLOR.get(mode, MODE_COLOR["unknown"])
            width = stroke_width(w)
            folium.PolyLine(
                locations=[NODE_COORDS[u], NODE_COORDS[v]],
                weight=width,
                color=color,
                tooltip=f"{u} ↔ {v} — {w} min ({mode})",
                popup=folium.Popup(f"<b>{NODE_NAMES.get(u,u)}</b> ↔ <b>{NODE_NAMES.get(v,v)}</b><br/>{w} minutes ({mode})", max_width=300)
            ).add_to(m)

    # highlight path (red)
    if highlight_path and len(highlight_path) >= 2:
        path_coords = [NODE_COORDS[n] for n in highlight_path if n in NODE_COORDS]
        folium.PolyLine(locations=path_coords, weight=6, color="red", opacity=0.9, tooltip="Shortest path").add_to(m)
        for i,node in enumerate(highlight_path):
            if node in NODE_COORDS:
                folium.CircleMarker(location=NODE_COORDS[node], radius=9, color="red", fill=True, fill_opacity=0.9,
                                    popup=f"{i+1}. {node}: {NODE_NAMES.get(node,node)}").add_to(m)

    # mark nodes: labeled with node letter (DivIcon). color differently if AP or bridge endpoint
    bridge_nodes = set()
    if bridges:
        for a,b in bridges:
            bridge_nodes.add(a); bridge_nodes.add(b)

    for node, (lat, lon) in NODE_COORDS.items():
        if node not in adj:
            continue
        html = f"<b>{node}</b>: {NODE_NAMES.get(node,node)}"
        # color logic
        bg_color = "white"
        if ap_list and node in ap_list:
            bg_color = "#FFA500"  # orange for AP
        if node in bridge_nodes:
            bg_color = "#800080"  # purple for bridge endpoints
        # DivIcon to show letter inside colored box
        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(html, max_width=300),
            tooltip=f"{node}: {NODE_NAMES.get(node,node)}",
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size:13px;
                    font-weight:bold;
                    color:black;
                    background:{bg_color};
                    border:1px solid #222;
                    border-radius:4px;
                    padding:4px 6px;
                    text-align:center;
                    min-width:20px;
                ">{node}</div>
            """)
        ).add_to(m)

    return m

# --------------------------
# Run analysis (with simulation if chosen)
# --------------------------
adj_for_run = ADJ
impact_text = ""
if failure_mode == "Remove Edge" and fail_edge:
    adj_for_run = remove_edge(ADJ, fail_edge[0], fail_edge[1])
    impact_text = f"Simulated removal of edge {fail_edge[0]} — {fail_edge[1]}"
elif failure_mode == "Remove Node" and fail_node:
    adj_for_run = remove_node(ADJ, fail_node)
    impact_text = f"Simulated removal of node {fail_node}: {NODE_NAMES.get(fail_node,fail_node)}"

ap_list_run, bridges_run = find_bridges_and_articulation_points(adj_for_run)

result_dist = {}
result_parent = {}
result_path: List[str] = []

if compute_button:
    if src not in adj_for_run or tgt not in adj_for_run:
        st.error("Source or target missing (removed in simulation). Choose different nodes or disable failure simulation.")
    else:
        result_dist, result_parent = dijkstra(adj_for_run, src)
        if result_dist.get(tgt, float("inf")) == float("inf"):
            st.error(f"No path found from {src} to {tgt}. {impact_text}")
        else:
            result_path = reconstruct_path(result_parent, tgt)
            baseline_dist, baseline_parent = dijkstra(ADJ, src)
            base_time = baseline_dist.get(tgt, float("inf"))
            run_time = result_dist.get(tgt, float("inf"))
            if base_time < float("inf"):
                delta = run_time - base_time
                if delta > 0:
                    impact_text = f"After simulation, travel time increased by {delta:.1f} min (was {base_time:.1f}m, now {run_time:.1f}m)."
                elif delta == 0:
                    impact_text = f"No change in shortest-path time (still {run_time:.1f} min)."
                else:
                    impact_text = f"Unexpected: travel time decreased by {-delta:.1f} min (now {run_time:.1f}m vs {base_time:.1f}m baseline)."
            else:
                impact_text = f"No baseline path found (baseline disconnected). Current path: {run_time:.1f} min."

# Build the folium map and embed it
with st.expander("Map & Visualization (interactive) — click to open"):
    if compute_button:
        fmap = build_map(adj_for_run, highlight_path=result_path, ap_list=ap_list_run if show_ap else None, bridges=bridges_run if show_bridges else None)
    else:
        fmap = build_map(ADJ, highlight_path=None, ap_list=base_ap if show_ap else None, bridges=base_bridges if show_bridges else None)
    map_html = fmap.get_root().render()
    components.html(map_html, height=700)

# --------------------------
# Results column
# --------------------------
st.markdown("---")
c1, c2 = st.columns([2,1])
with c1:
    st.subheader("Graph Details & Shortest Path Results")
    st.write(f"Nodes (count): **{len(ADJ)}** — Edges (undirected count): **{sum(len(lst) for lst in ADJ.values())//2}**")
    if compute_button:
        if result_dist and result_path:
            st.success(f"Shortest time {src} → {tgt}: **{result_dist[tgt]:.1f} minutes**")
            st.info("Sequence of stops (with labels):")
            seq = " → ".join([f"{n} ({NODE_NAMES.get(n,n)})" for n in result_path])
            st.write(seq)
            edge_rows = []
            for i in range(len(result_path)-1):
                u = result_path[i]; v = result_path[i+1]
                w = next((wt for nbr,wt,mode in ADJ[u] if nbr==v), None)
                mode = next((mode for nbr,wt,mode in ADJ[u] if nbr==v), "unknown")
                edge_rows.append({"from": f"{u} ({NODE_NAMES.get(u)})", "to": f"{v} ({NODE_NAMES.get(v)})", "time_min": w, "mode": mode})
            st.table(pd.DataFrame(edge_rows))
            if impact_text:
                st.write(f"**Simulation impact:** {impact_text}")
        else:
            st.info("Press 'Compute shortest path & analyze' to run Dijkstra and critical-point analysis.")
with c2:
    st.subheader("Critical Points")
    st.write("Articulation Points (APs):")
    if ap_list_run:
        for a in ap_list_run:
            st.write(f"- {a}: {NODE_NAMES.get(a,a)}")
    else:
        st.write("None found.")
    st.write("Bridges (cut edges):")
    if bridges_run:
        for u,v in bridges_run:
            st.write(f"- {u} — {v}: {NODE_NAMES.get(u,u)} ↔ {NODE_NAMES.get(v,v)}")
    else:
        st.write("None found.")

# --------------------------
# Adjacency & Export
# --------------------------
st.markdown("---")
st.subheader("Adjacency List (JSON)")
adj_serial = {u: [[v,w,mode] for v,w,mode in ADJ[u]] for u in ADJ}
st.code(json.dumps(adj_serial, indent=2))

st.markdown("#### Adjacency Matrix (sample)")
mat_df = adjacency_matrix(ADJ)
st.dataframe(mat_df.astype(object).replace(np.inf, "∞").head(12))

st.markdown("---")
st.subheader("Export")
adj_bytes = json.dumps(adj_serial, indent=2).encode("utf-8")
st.download_button("Download adjacency JSON", data=adj_bytes, file_name="baltimore_adjacency.json", mime="application/json")

# Project report text
report_lines = []
report_lines.append("Baltimore Inner Harbor Transit Graph — Project Report")
report_lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC")
report_lines.append("\nNodes:")
for k in sorted(ADJ.keys()):
    report_lines.append(f"{k}: {NODE_NAMES.get(k,k)} ({NODE_COORDS.get(k,('n/a','n/a'))})")
report_lines.append("\nJustification of data structure:")
report_lines.append("Adjacency list chosen due to sparse nature of city transit graph (each node connects to few neighbors).")
report_lines.append("\nCritical points (Articulation Points):")
if base_ap:
    for a in base_ap:
        report_lines.append(f"- {a}: {NODE_NAMES.get(a,a)}")
else:
    report_lines.append("None found.")
report_lines.append("\nBridges (cut edges):")
if base_bridges:
    for u,v in base_bridges:
        report_lines.append(f"- {u} — {v} : {NODE_NAMES.get(u,u)} ↔ {NODE_NAMES.get(v,v)}")
else:
    report_lines.append("None found.")
bd, bp = dijkstra(ADJ, "P")
for tgt_ex in ["A","F"]:
    if bd.get(tgt_ex, float("inf")) < float("inf"):
        path_ex = reconstruct_path(bp, tgt_ex)
        report_lines.append(f"P → {tgt_ex}: {bd[tgt_ex]:.1f} min via {'->'.join(path_ex)}")
    else:
        report_lines.append(f"P → {tgt_ex}: no path found")

report_txt = "\n".join(report_lines)
st.download_button("Download Project Report (txt)", data=report_txt.encode("utf-8"), file_name="project_report.txt", mime="text/plain")

st.markdown("---")
st.caption("Tips: Replace DEFAULT_ADJ weights with measured times from Google Maps for higher accuracy. Water taxi edges include wait time estimates.")
