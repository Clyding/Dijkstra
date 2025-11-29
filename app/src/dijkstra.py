# app.py
import streamlit as st
import json
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import textwrap

st.set_page_config(page_title="Baltimore Transit Graph — Dijkstra + Critical Points", layout="wide")

st.title("Baltimore Inner Harbor Transit Graph — Dijkstra, Bridges & Articulation Points")
st.markdown(
    """
    This app models locations in and around Baltimore's Inner Harbor as a weighted, undirected graph.
    Nodes = locations. Edge weights = travel time in minutes (includes reasonable wait times for transit).
    """
)

NODES = {
    "P": "Penn Station (Transit Hub)",
    "A": "National Aquarium",
    "F": "Fort McHenry",
    "B": "M&T Bank Stadium",
    "C": "Convention Center (Light Rail Stop)",
    "F": "Fells Point",
    "E": "Federal Hill",
    "G": "Harbor East",
    "H": "Harbor Point",
    "I": "Inner Harbor (Harborplace)",
    "J": "Jonestown",
    "K": "Locust Point / Ferry Area",
    "L": "Little Italy",
    "M": "Mount Vernon",
    "O": "Oriole Park (Camden Yards)",
    "Q": "Lexington Market / Downtown",
    "R": "Ridgely's Delight",
    "O": "Otterbein",
    "T": "Inner Harbor Water Taxi Dock",
}

DEFAULT_ADJ = {
    "P": [("M", 6, "walk"), ("L", 8, "walk"), ("C", 12, "walk") , ("S",10,"walk")],
    "M": [("P", 6, "walk"), ("B", 9, "light_rail"), ("C", 7, "light_rail")],
    "L": [("P", 8, "walk"), ("Q", 10, "walk")],
    "Q": [("L", 10, "walk"), ("C", 7, "walk")],
    "C": [("A", 3, "walk"), ("O", 2, "walk"), ("B", 10, "walk"), ("D", 9, "walk"),
          ("E", 7, "walk"), ("G", 8, "walk"), ("I",5,"walk"), ("R",5,"walk"),("S",8,"walk"),("T",7,"walk"),("P",12,"walk")],
    "A": [("C", 3, "walk")],
    "O": [("C", 2, "walk"), ("F", 20, "water_taxi")], 
    "F": [("O", 20, "water_taxi")],
    "B": [("M", 9, "light_rail"), ("C", 10, "walk"), ("J", 4, "walk"), ("K", 6, "walk")],
    "J": [("B", 4, "walk"), ("K", 6, "walk")],
    "K": [("B", 6, "walk"), ("J", 6, "walk"), ("I", 12, "walk"), ("E",9,"walk"),("C",14,"walk")],
    "D": [("C", 9, "walk"), ("H", 6, "walk"), ("R",6,"walk")],
    "H": [("D", 6, "walk"), ("G", 5, "walk"), ("C",6,"walk")],
    "G": [("H", 5, "walk"), ("I", 6, "walk"), ("C",8,"walk")],
    "I": [("G", 6, "walk"), ("C",5,"walk"), ("K",12,"walk")],
    "E": [("C", 7, "walk"), ("K", 9, "walk")],
    "R": [("C",5,"walk"), ("D",6,"walk")],
    "S": [("P",10,"walk"), ("C",8,"walk")],
    "T": [("C",7,"walk")] 
}

st.sidebar.header("Data / Graph Options")
use_default = st.sidebar.checkbox("Use built-in example graph (recommended)", value=True)

adj_text = ""
adj_data = DEFAULT_ADJ
if not use_default:
    st.sidebar.write("Paste an adjacency JSON (format: dict of node -> list of (neighbor, weight, mode))")
    adj_text = st.sidebar.text_area("Adjacency JSON", height=200, value=json.dumps(DEFAULT_ADJ, indent=2))
    try:
        adj_data = json.loads(adj_text)
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        adj_data = DEFAULT_ADJ

def normalize_undirected(adj: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    g = {}
    for u, nbrs in adj.items():
        g.setdefault(u, [])
        for entry in nbrs:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                v, w = entry[0], entry[1]
                mode = entry[2] if len(entry) >= 3 else "walk"
                g[u].append((v, float(w), mode))
                g.setdefault(v, [])
                g[v].append((u, float(w), mode))
            else:
                raise ValueError("Adj list entries must be (node, weight [, mode])")
    for u in list(g.keys()):
        seen = set()
        dedup = []
        for v,w,mode in g[u]:
            key = (v, float(w), mode)
            if key not in seen:
                dedup.append((v,w,mode))
                seen.add(key)
        g[u] = dedup
    return g

try:
    ADJ = normalize_undirected(adj_data)
except Exception as e:
    st.error("Error parsing adjacency data, falling back to built-in.")
    ADJ = normalize_undirected(DEFAULT_ADJ)

ALL_NODES = sorted(ADJ.keys())


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
    pq = []
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
    if parent[target] is None and target not in parent:
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
    disc = {}
    low = {}
    parent = {}
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

def remove_edge(adj: Dict[str, List[Tuple]], u: str, v: str):
    newadj = {k: [tuple(x) for x in lst] for k,lst in adj.items() for k in [k]}  
    for a,b in [(u,v),(v,u)]:
        if a in newadj:
            newadj[a] = [t for t in newadj[a] if t[0] != b]
    return newadj


col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Query: Shortest Path (Dijkstra)")
    src = st.selectbox("Source node", ALL_NODES, index=ALL_NODES.index("P") if "P" in ALL_NODES else 0)
    tgt = st.selectbox("Target node", ALL_NODES, index=ALL_NODES.index("A") if "A" in ALL_NODES else 1)
    run_btn = st.button("Compute shortest path")
    st.markdown("---")
    st.subheader("Simulate failure")
    failure_type = st.selectbox("Fail:", ["None", "Remove Edge", "Remove Node"], index=0)
    fail_edge = None
    fail_node = None
    if failure_type == "Remove Edge":
        edges_flat = []
        for u in sorted(ADJ.keys()):
            for v,w,mode in ADJ[u]:
                if u < v:
                    edges_flat.append(f"{u} - {v} ({w}m, {mode})")
        chosen_edge = st.selectbox("Select edge to remove", edges_flat)
        u_v = chosen_edge.split()[0:3]
        uu = chosen_edge.split()[0]
        vv = chosen_edge.split()[2]
        fail_edge = (uu, vv)
    elif failure_type == "Remove Node":
        fail_node = st.selectbox("Node to remove", ALL_NODES)

with col2:
    st.subheader("Graph Summary & Diagnostics")
    st.write(f"Nodes: **{len(ALL_NODES)}**")
    total_edges = sum(len(lst) for lst in ADJ.values()) // 2
    st.write(f"Edges (undirected count): **{total_edges}**")
    st.write("Note: edge weights are minutes (includes waits for transit edges).")

    st.markdown("#### Adjacency List (sample)")
    rows = []
    for u in sorted(ADJ.keys()):
        for v,w,mode in ADJ[u]:
            rows.append({"from":u, "to":v, "time_min":w, "mode":mode})
    df = pd.DataFrame(rows)
    st.dataframe(df.head(200))

    st.markdown("#### Adjacency Matrix (infinite means no direct route)")
    mat_df = adjacency_matrix(ADJ)
    st.dataframe(mat_df.astype(object).replace(np.inf, "∞").head(12))

ap_list, bridges = find_bridges_and_articulation_points(ADJ)

st.markdown("---")
st.subheader("Critical Point Analysis (Computed)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Articulation Points (Cut Vertices)**")
    if ap_list:
        for a in ap_list:
            st.write(f"- {a}: {NODES.get(a,a)}")
    else:
        st.write("No articulation points found (graph highly connected).")
with c2:
    st.markdown("**Bridges (Cut Edges)**")
    if bridges:
        for u,v in bridges:
            st.write(f"- {u} — {v}  ({NODES.get(u,u)}  ↔  {NODES.get(v,v)})")
    else:
        st.write("No bridges found.")

if run_btn:
    adj_for_run = ADJ
    fail_report = ""
    if failure_type == "Remove Edge" and fail_edge is not None:
        u,v = fail_edge
        adj_for_run = remove_edge(ADJ, u, v)
        fail_report = f"Edge removed: {u} — {v}"
    elif failure_type == "Remove Node" and fail_node is not None:
        adj_for_run = {k:[(x,y,z) for x,y,z in v if x != fail_node] for k,v in ADJ.items() if k != fail_node}
        fail_report = f"Node removed: {fail_node} — {NODES.get(fail_node,fail_node)}"

    dist, parent = dijkstra(adj_for_run, src)
    if dist.get(tgt, float("inf")) == float("inf"):
        st.error(f"No path found from {src} to {tgt} (graph disconnected after failure?) {fail_report}")
    else:
        path = reconstruct_path(parent, tgt)
        st.success(f"Shortest time from **{src}** to **{tgt}**: **{dist[tgt]:.1f} minutes**")
        readable_path = " → ".join([f"{n} ({NODES.get(n,n)})" for n in path])
        st.write("Sequence of stops:")
        st.info(readable_path)

        edge_rows = []
        for i in range(len(path)-1):
            u = path[i]; v = path[i+1]
            w = next((wt for nbr,wt,mode in ADJ[u] if nbr==v), None)
            mode = next((mode for nbr,wt,mode in ADJ[u] if nbr==v), "unknown")
            edge_rows.append({"from":u, "to":v, "time_min": w, "mode": mode})
        st.table(pd.DataFrame(edge_rows))

        if failure_type != "None":
            base_dist, base_parent = dijkstra(ADJ, src)
            if base_dist.get(tgt, float("inf")) == float("inf"):
                st.warning("Baseline (no failure) had no path — strange.")
            else:
                delta = dist[tgt] - base_dist[tgt]
                if delta > 0:
                    st.warning(f"Impact of failure: travel time increased by **{delta:.1f} minutes** (was {base_dist[tgt]:.1f} min).")
                elif delta == 0:
                    st.info("Impact of failure: no change to shortest-path travel time.")
                else:
                    st.success(f"Impact: travel time decreased by {-delta:.1f} minutes (unexpected; check graph).")

st.markdown("---")
st.subheader("Export / Save")

def adjacency_json_bytes(adj: Dict[str,List[Tuple]]):
    serial = {}
    for u,lst in adj.items():
        serial[u] = [[v, w, mode] for v,w,mode in lst]
    return json.dumps(serial, indent=2).encode("utf-8")

colx, coly = st.columns(2)
with colx:
    st.download_button("Download adjacency JSON", data=adjacency_json_bytes(ADJ), file_name="baltimore_adjacency.json", mime="application/json")

with coly:
    report_lines = []
    report_lines.append("Baltimore Inner Harbor Transit Graph — Project Report\n")
    report_lines.append("Nodes and short names:\n")
    for k in sorted(ADJ.keys()):
        report_lines.append(f"{k}: {NODES.get(k,k)}\n")
    report_lines.append("\nJustification of data structure:\n")
    report_lines.append(
        "We chose an adjacency list representation. City transit graphs are typically sparse: "
        "each intersection or point connects to a few neighboring points only, not to every other node. "
        "Adjacency lists therefore use far less memory and allow efficient iteration over neighbors. "
        "They also pair well with Dijkstra's priority-queue implementation (O(E + V log V)).\n"
    )
    report_lines.append("\nCritical points (Articulation Points):\n")
    if ap_list:
        for a in ap_list:
            report_lines.append(f"- {a}: {NODES.get(a,a)}\n")
    else:
        report_lines.append("None found.\n")
    report_lines.append("\nBridges (edges whose removal disconnects network):\n")
    if bridges:
        for u,v in bridges:
            report_lines.append(f"- {u} — {v} ({NODES.get(u,u)} ↔ {NODES.get(v,v)})\n")
    else:
        report_lines.append("None found.\n")
    report_lines.append("\nExample shortest-path scenario:\n")
    ex_src = "P"
    for ex_tgt in ["A","F"]:
        dd, pp = dijkstra(ADJ, ex_src)
        if dd.get(ex_tgt, float("inf")) < float("inf"):
            path = reconstruct_path(pp, ex_tgt)
            report_lines.append(f"- {ex_src} → {ex_tgt}: {dd[ex_tgt]:.1f} min via {'->'.join(path)}\n")
        else:
            report_lines.append(f"- {ex_src} → {ex_tgt}: no path found\n")
    report_lines.append("\nSimulation advice: To measure resilience, remove a bridge edge (e.g., a water taxi link) and recompute paths. "
                        "The increase in shortest-path time quantifies the loss of network reliability.\n")
    report_txt = "\n".join(report_lines)

    st.download_button("Download Project Report (txt)", data=report_txt.encode("utf-8"), file_name="project_report.txt", mime="text/plain")

st.markdown("---")
st.subheader("Network visualization (layout is schematic)")
G = nx.Graph()
for u, lst in ADJ.items():
    G.add_node(u, label=NODES.get(u,u))
for u, lst in ADJ.items():
    for v,w,mode in lst:
        if not G.has_edge(u,v):
            G.add_edge(u, v, weight=w, mode=mode)

fig, ax = plt.subplots(figsize=(10,6))
pos = nx.spring_layout(G, seed=42, k=0.6)
nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax)
nx.draw_networkx_labels(G, pos, labels={n:n for n in G.nodes()}, font_size=8)
nx.draw_networkx_edges(G, pos, ax=ax)
edge_labels = {(u,v): f"{d['weight']:.0f}m" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
ax.set_axis_off()
st.pyplot(fig)

st.caption("Schematic network layout (spring layout). Replace with geospatial coordinates for map-accurate visualization.")
