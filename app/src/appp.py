import streamlit as st
import json
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components
import datetime
import zipfile
import io
import requests
from folium.plugins import AntPath, PolyLineTextPath, Draw

try:
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except Exception:
    STREAMLIT_FOLIUM_AVAILABLE = False

st.set_page_config(page_title="Baltimore Transit — Geo Graph + Dijkstra + Critical Points", layout="wide")
st.title("Baltimore Inner Harbor — Geo Graph + Dijkstra + Critical Points")

NODE_NAMES = {
    "P": "Penn Station (Transit Hub)","A": "National Aquarium","F": "Fort McHenry",
    "B": "M&T Bank Stadium","C": "Convention Center (Light Rail Stop)","D": "Fells Point",
    "E": "Federal Hill","G": "Harbor East","H": "Harbor Point","I": "Inner Harbor (Harborplace)",
    "J": "Jonestown","K": "Locust Point / Ferry Area","L": "Little Italy","M": "Mount Vernon",
    "O": "Oriole Park / Camden Yards (alt)","Q": "Lexington Market / Downtown","R": "Ridgely's Delight",
    "S": "Otterbein","T": "Inner Harbor Water Taxi Dock"
}

NODE_COORDS = {
    "P": (39.3079, -76.6157),"A": (39.2857, -76.6081),"F": (39.2813, -76.5803),
    "B": (39.2839, -76.6217),"C": (39.2851, -76.6187),"D": (39.2830, -76.6050),
    "E": (39.2789, -76.6120),"G": (39.2833, -76.6026),"H": (39.2784, -76.6000),
    "I": (39.2850, -76.6132),"J": (39.2913, -76.6052),"K": (39.2708, -76.5930),
    "L": (39.2867, -76.6030),"M": (39.3009, -76.6158),"O": (39.2821, -76.6188),
    "Q": (39.2911, -76.6208),"R": (39.2847, -76.6220),"S": (39.2930, -76.6140),
    "T": (39.2844, -76.6105)
}

DEFAULT_ADJ = {
    "P": [("M", 6, "walk"), ("L", 8, "walk"), ("C", 12, "walk"), ("S", 10, "walk")],
    "M": [("P", 6, "walk"), ("B", 9, "light_rail"), ("C", 7, "light_rail")],
    "L": [("P", 8, "walk"), ("Q", 10, "walk"), ("D", 6, "walk")],
    "Q": [("L", 10, "walk"), ("C", 7, "walk")],
    "C": [("A", 3, "walk"), ("T", 2, "walk"), ("B", 10, "walk"), ("D", 9, "walk"),
          ("E", 7, "walk"), ("G", 8, "walk"), ("I", 5, "walk"), ("R", 5, "walk"), ("S", 8, "walk"), ("P", 12, "walk")],
    "A": [("C", 3, "walk")],"T": [("C", 2, "walk"), ("F", 20, "water_taxi")],
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

def normalize_undirected(adj: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    g: Dict[str, List[Tuple[str,float,str]]] = {}
    for u, nbrs in adj.items():
        g.setdefault(u, [])
        for entry in nbrs:
            v = entry[0]; w = float(entry[1]); mode = entry[2] if len(entry)>=3 else "walk"
            g[u].append((v,w,mode))
            g.setdefault(v, []); g[v].append((u,w,mode))
    for u in list(g.keys()):
        seen = set(); dedup=[]
        for v,w,mode in g[u]:
            k=(v,float(w),mode)
            if k not in seen:
                dedup.append((v,w,mode)); seen.add(k)
        g[u]=dedup
    return g

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

NODES = sorted(ADJ.keys())
src = st.sidebar.selectbox("Source", NODES, index=NODES.index("P") if "P" in NODES else 0)
tgt = st.sidebar.selectbox("Target", NODES, index=NODES.index("A") if "A" in NODES else 0)
compute_button = st.sidebar.button("Compute shortest path & analyze")

failure_mode = st.sidebar.selectbox("Simulate failure:", ["None", "Remove Edge", "Remove Node"])
fail_edge = None; fail_node = None
if failure_mode=="Remove Edge":
    edges_flat=[]; seen=set()
    for u in sorted(ADJ.keys()):
        for v,w,mode in ADJ[u]:
            a,b=sorted((u,v))
            if (a,b) not in seen:
                seen.add((a,b)); edges_flat.append(f"{a} - {b} ({w}m, {mode})")
    chosen = st.sidebar.selectbox("Edge to remove", edges_flat)
    uu = chosen.split()[0]; vv = chosen.split()[2]; fail_edge=(uu,vv)
elif failure_mode=="Remove Node":
    fail_node = st.sidebar.selectbox("Node to remove", NODES)

show_ap = st.sidebar.checkbox("Show articulation points", True, key="show_ap")
show_bridges = st.sidebar.checkbox("Show bridges", True, key="show_bridges")
show_all_edges = st.sidebar.checkbox("Show all edges", True, key="show_all_edges")
animate_path = st.sidebar.checkbox("Animate shortest-path (AntPath)", True, key="animate_path")
show_arrows = st.sidebar.checkbox("Show direction arrows on edges", True, key="show_arrows")
st.sidebar.header("GTFS / Live Data")
uploaded_gtfs = st.sidebar.file_uploader("Upload GTFS ZIP (optional)", type=["zip"])
gmaps_key = st.sidebar.text_input("Google Maps API Key (optional, for live times)", type="password")
use_gmaps = bool(gmaps_key.strip())
use_live_gmaps = st.sidebar.checkbox("Use Google Maps to refresh edge weights (for shown edges)", value=False) and use_gmaps

def adjacency_matrix(adj: Dict[str, List[Tuple]]) -> pd.DataFrame:
    nodes = sorted(adj.keys()); n=len(nodes)
    mat = np.full((n,n), np.inf); idx={node:i for i,node in enumerate(nodes)}
    for u in nodes:
        mat[idx[u], idx[u]] = 0.0
        for v,w,mode in adj[u]:
            mat[idx[u], idx[v]] = w
    return pd.DataFrame(mat, index=nodes, columns=nodes)

def dijkstra(adj: Dict[str, List[Tuple]], source: str):
    dist={node:float('inf') for node in adj}; parent={node:None for node in adj}
    dist[source]=0.0; pq=[]; heappush(pq,(0.0,source))
    while pq:
        d,u=heappop(pq)
        if d>dist[u]: continue
        for v,w,mode in adj[u]:
            nd=d+float(w)
            if nd<dist[v]:
                dist[v]=nd; parent[v]=u; heappush(pq,(nd,v))
    return dist,parent

def reconstruct_path(parent, target):
    if target not in parent: return []
    path=[]; cur=target
    while cur is not None:
        path.append(cur); cur=parent.get(cur)
    path.reverse(); return path

def find_bridges_and_articulation_points(adj):
    time=0; disc={}; low={}; parent={}; ap_set=set(); bridges=[]
    for u in adj:
        disc[u]=-1; low[u]=-1; parent[u]=None
    def dfs(u):
        nonlocal time
        disc[u]=time; low[u]=time; time+=1
        children=0
        for v,w,mode in adj[u]:
            if disc[v]==-1:
                parent[v]=u; children+=1; dfs(v)
                low[u]=min(low[u], low[v])
                if parent[u] is None and children>1: ap_set.add(u)
                if parent[u] is not None and low[v] >= disc[u]: ap_set.add(u)
                if low[v] > disc[u]: bridges.append((u,v))
            elif parent[u] != v:
                low[u]=min(low[u], disc[v])
    for node in adj.keys():
        if disc[node]==-1: dfs(node)
    bridges_norm=[tuple(sorted((u,v))) for u,v in bridges]; bridges_norm=sorted(list(set(bridges_norm)))
    return sorted(list(ap_set)), bridges_norm

def parse_gtfs_add_edges(gtfs_bytes, adj):
    z = zipfile.ZipFile(io.BytesIO(gtfs_bytes))
    files = z.namelist()
    if "stops.txt" not in files or "stop_times.txt" not in files:
        st.warning("GTFS must include stops.txt and stop_times.txt to auto-create edges.")
        return adj
    stops = pd.read_csv(z.open("stops.txt"))
    stop_times = pd.read_csv(z.open("stop_times.txt"))
    stop_map = {}
    if "stop_id" in stops.columns and "stop_lat" in stops.columns and "stop_lon" in stops.columns:
        for _,row in stops.iterrows():
            stop_map[row["stop_id"]] = (row["stop_lat"], row["stop_lon"], row.get("stop_name", ""))
    for trip_id, grp in stop_times.groupby("trip_id"):
        grp_sorted = grp.sort_values("stop_sequence")
        seq = list(grp_sorted["stop_id"])
        for i in range(len(seq)-1):
            s1, s2 = seq[i], seq[i+1]
            if s1 in stop_map and s2 in stop_map:
                lat1, lon1, name1 = stop_map[s1]
                lat2, lon2, name2 = stop_map[s2]
                for sid,lat,lon,name in [(s1,lat1,lon1,name1),(s2,lat2,lon2,name2)]:
                    if sid not in adj:
                        adj[sid]=[(None,0,"walk")]
                        NODE_COORDS[sid] = (lat, lon)
                        NODE_NAMES[sid] = name if name else sid
                try:
                    t1 = grp_sorted.iloc[i]["departure_time"]
                    t2 = grp_sorted.iloc[i+1]["arrival_time"]
                    def to_minutes(t):
                        h,m,s = map(int, t.split(":"))
                        return h*60 + m + s/60.0
                    w = max(1.0, to_minutes(t2) - to_minutes(t1))
                except Exception:
                    w = 5.0
                adj.setdefault(seq[i], [])
                adj.setdefault(seq[i+1], [])
                adj[seq[i]].append((seq[i+1], w, "gtfs"))
                adj[seq[i+1]].append((seq[i], w, "gtfs"))
    st.success("GTFS parsed and edges added (stop_ids used as node labels).")
    return adj

def gmaps_travel_time(origin_latlon, dest_latlon, api_key):
    try:
        origin = f"{origin_latlon[0]},{origin_latlon[1]}"
        dest = f"{dest_latlon[0]},{dest_latlon[1]}"
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {"origin": origin, "destination": dest, "key": api_key, "mode": "transit"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("routes"):
            total_sec = sum(leg.get("duration", {}).get("value", 0) for leg in data["routes"][0]["legs"])
            return total_sec / 60.0
        else:
            params["mode"] = "walking"
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("routes"):
                total_sec = sum(leg.get("duration", {}).get("value", 0) for leg in data["routes"][0]["legs"])
                return total_sec / 60.0
    except Exception:
        return None
    return None

if uploaded_gtfs is not None:
    try:
        gtfs_bytes = uploaded_gtfs.read()
        ADJ = parse_gtfs_add_edges(gtfs_bytes, ADJ)
        NODES = sorted(ADJ.keys())
    except Exception as e:
        st.sidebar.error(f"Failed to parse GTFS: {e}")

new_node_label = st.sidebar.text_input("New node label (optional)", value="")
new_node_name = st.sidebar.text_input("New node name (optional)", value="")

def remove_edge(adj, u, v):
    newadj = {k:[(x,y,z) for x,y,z in lst] for k,lst in adj.items() for k in [k]}
    for a,b in [(u,v),(v,u)]:
        if a in newadj: newadj[a] = [t for t in newadj[a] if t[0] != b]
    return newadj

def remove_node(adj, node):
    newadj={}
    for u,lst in adj.items():
        if u==node: continue
        newadj[u]=[(v,w,mode) for v,w,mode in lst if v!=node]
    return newadj

base_ap, base_bridges = find_bridges_and_articulation_points(ADJ)

MODE_COLOR = {"walk": "blue", "light_rail": "red", "water_taxi": "cyan", "gtfs": "purple", "unknown": "gray"}

def build_map(adj: Dict[str, List[Tuple]], highlight_path: Optional[List[str]]=None,
              ap_list: Optional[List[str]]=None, bridges: Optional[List[Tuple[str,str]]]=None,
              arrows: bool=True, animate: bool=True):
    center=[39.2904, -76.6122]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    Draw(export=True, filename='data.geojson').add_to(m)
    weights = [w for u in adj for _,w,_ in adj[u]] or [1]
    min_w, max_w = min(weights), max(weights)
    def stroke(w):
        if max_w==min_w: return 3
        return 2 + 6*((w-min_w)/(max_w-min_w))
    seen=set()
    for u in adj:
        if u not in NODE_COORDS: continue
        for v,w,mode in adj[u]:
            a,b=sorted((u,v))
            if (a,b) in seen: continue
            seen.add((a,b))
            if v not in NODE_COORDS: continue
            color = MODE_COLOR.get(mode, "gray")
            sw = stroke(w)
            line = folium.PolyLine(locations=[NODE_COORDS[u], NODE_COORDS[v]], weight=sw, color=color,
                                   tooltip=f"{u} ↔ {v} — {w} min ({mode})",
                                   popup=folium.Popup(f"<b>{NODE_NAMES.get(u,u)}</b> ↔ <b>{NODE_NAMES.get(v,v)}</b><br/>{w} minutes ({mode})", max_width=300))
            line.add_to(m)
            if arrows:
                try:
                    PolyLineTextPath(line, " ➤ ", repeat=True, offset=8, attributes={'fill':'black','font-weight':'bold','font-size':'14'}).add_to(m)
                except Exception:
                    pass
    if highlight_path and len(highlight_path)>=2:
        path_coords = [NODE_COORDS[n] for n in highlight_path if n in NODE_COORDS]
        if animate:
            try:
                AntPath(locations=path_coords, color="yellow", weight=6, delay=1000).add_to(m)
            except Exception:
                folium.PolyLine(locations=path_coords, weight=6, color="yellow").add_to(m)
        else:
            folium.PolyLine(locations=path_coords, weight=6, color="yellow").add_to(m)
        for i,node in enumerate(highlight_path):
            if node in NODE_COORDS:
                folium.CircleMarker(location=NODE_COORDS[node], radius=7, color="orange", fill=True,
                                    popup=f"{i+1}. {node}: {NODE_NAMES.get(node,node)}").add_to(m)
    bridge_nodes = set()
    if bridges:
        for a,b in bridges: bridge_nodes.add(a); bridge_nodes.add(b)
    for node,(lat,lon) in NODE_COORDS.items():
        if node not in adj: continue
        bg='white'
        if ap_list and node in ap_list: bg='#FFA500'
        if node in bridge_nodes: bg='#800080'
        folium.Marker(location=(lat,lon), popup=f"{node}: {NODE_NAMES.get(node,node)}",
                      tooltip=f"{node}: {NODE_NAMES.get(node,node)}",
                      icon=folium.DivIcon(html=f"""<div style="font-size:13px;font-weight:bold;color:black;background:{bg};
                                              border:1px solid #222;border-radius:4px;padding:4px 6px;text-align:center;min-width:20px">{node}</div>""")
                     ).add_to(m)
    return m

adj_for_run = ADJ
impact_text = ""
if failure_mode=="Remove Edge" and fail_edge:
    adj_for_run = remove_edge(ADJ, fail_edge[0], fail_edge[1])
    impact_text = f"Simulated removal of edge {fail_edge[0]} — {fail_edge[1]}"
elif failure_mode=="Remove Node" and fail_node:
    adj_for_run = remove_node(ADJ, fail_node)
    impact_text = f"Simulated removal of node {fail_node}: {NODE_NAMES.get(fail_node,fail_node)}"

ap_list_run, bridges_run = find_bridges_and_articulation_points(adj_for_run)
result_dist = {}; result_parent = {}; result_path: List[str]=[]

if compute_button:
    if use_live_gmaps and use_gmaps:
        updated_adj = {u:[] for u in adj_for_run}
        seen=set()
        for u in adj_for_run:
            for v,w,mode in adj_for_run[u]:
                a,b = sorted((u,v))
                if (a,b) in seen: continue
                seen.add((a,b))
                if u in NODE_COORDS and v in NODE_COORDS:
                    new_w = gmaps_travel_time(NODE_COORDS[u], NODE_COORDS[v], gmaps_key) or w
                    updated_adj[u].append((v,new_w,mode)); updated_adj[v].append((u,new_w,mode))
                else:
                    updated_adj[u].append((v,w,mode)); updated_adj[v].append((u,w,mode))
        adj_for_run = updated_adj
    if src not in adj_for_run or tgt not in adj_for_run:
        st.error("Source or target missing (removed in simulation).")
    else:
        result_dist, result_parent = dijkstra(adj_for_run, src)
        if result_dist.get(tgt, float('inf')) == float('inf'):
            st.error(f"No path found from {src} to {tgt}. {impact_text}")
        else:
            result_path = reconstruct_path(result_parent, tgt)
            base_dist, _ = dijkstra(ADJ, src)
            if base_dist.get(tgt, float('inf'))<float('inf'):
                delta = result_dist[tgt] - base_dist[tgt]
                if delta>0: impact_text = f"Travel time increased by {delta:.1f} min vs baseline."
                elif delta==0: impact_text = f"No change vs baseline ({result_dist[tgt]:.1f} min)."
                else: impact_text = f"Travel time decreased by {-delta:.1f} min vs baseline."

fmap = build_map(adj_for_run, highlight_path=result_path if compute_button else None,
                 ap_list=ap_list_run if show_ap else None,
                 bridges=bridges_run if show_bridges else None,
                 arrows=show_arrows, animate=animate_path)

if STREAMLIT_FOLIUM_AVAILABLE:
    map_data = st_folium(fmap, width=1000, height=700, returned_objects=["last_clicked", "all_drawings"])
    last_click = map_data.get("last_clicked")
    if last_click:
        lat = last_click["lat"]; lon = last_click["lng"]
        if st.sidebar.button("Add node at clicked location"):
            label = new_node_label.strip() or f"N{len(NODE_COORDS)+1}"
            name = new_node_name.strip() or label
            NODE_COORDS[label] = (lat, lon)
            NODE_NAMES[label] = name
            ADJ.setdefault(label, [])
            st.experimental_rerun()
else:
    map_html = fmap.get_root().render()
    components.html(map_html, height=700)

st.markdown("---")
c1, c2 = st.columns([2,1])
with c1:
    st.subheader("Graph Details & Shortest Path Results")
    st.write(f"Nodes (count): **{len(ADJ)}** — Edges (undirected count): **{sum(len(lst) for lst in ADJ.values())//2}**")
    if compute_button and result_dist and result_path:
        st.success(f"Shortest time {src} → {tgt}: **{result_dist[tgt]:.1f} minutes**")
        st.info("Sequence of stops:")
        st.write(" → ".join([f"{n} ({NODE_NAMES.get(n,n)})" for n in result_path]))
        rows=[]
        for i in range(len(result_path)-1):
            u=result_path[i]; v=result_path[i+1]
            w = next((wt for nbr,wt,mode in ADJ[u] if nbr==v), None)
            mode = next((m for nbr,wt,m in ADJ[u] if nbr==v), "unknown")
            rows.append({"from":f"{u} ({NODE_NAMES.get(u)})", "to":f"{v} ({NODE_NAMES.get(v)})", "time_min":w, "mode":mode})
        st.table(pd.DataFrame(rows))
        if impact_text: st.write(f"**Impact**: {impact_text}")
    else:
        st.info("Press 'Compute shortest path & analyze' to run Dijkstra and show animated path.")

with c2:
    st.subheader("Critical Points")
    st.write("Articulation Points (APs):")
    if ap_list_run:
        for a in ap_list_run: st.write(f"- {a}: {NODE_NAMES.get(a,a)}")
    else: st.write("None found.")
    st.write("Bridges (cut edges):")
    if bridges_run:
        for u,v in bridges_run: st.write(f"- {u} — {v}: {NODE_NAMES.get(u,u)} ↔ {NODE_NAMES.get(v,v)}")
    else: st.write("None found.")

st.markdown("---")
# st.subheader("Adjacency List (JSON)")
adj_serial = {u: [[v,w,mode] for v,w,mode in ADJ[u]] for u in ADJ}
# st.code(json.dumps(adj_serial, indent=2))

st.markdown("#### Adjacency Matrix (sample)")
mat_df = adjacency_matrix(ADJ)
st.dataframe(mat_df.astype(object).replace(np.inf, "∞").head(12))

st.markdown("---")
st.subheader("Export")
st.download_button("Download adjacency JSON", data=json.dumps(adj_serial, indent=2).encode("utf-8"),
                   file_name="baltimore_adjacency.json", mime="application/json")

report_lines=[]
report_lines.append("Baltimore Inner Harbor Transit Graph — Project Report")
report_lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC")
report_lines.append("\nNodes:")
for k in sorted(ADJ.keys()):
    report_lines.append(f"{k}: {NODE_NAMES.get(k,k)} ({NODE_COORDS.get(k,('n/a','n/a'))})")
report_lines.append("\nJustification: adjacency list (sparse graph).")
report_lines.append("\nCritical points:")
if base_ap:
    for a in base_ap: report_lines.append(f"- {a}: {NODE_NAMES.get(a,a)}")
else: report_lines.append("None found.")
if base_bridges:
    for u,v in base_bridges: report_lines.append(f"- {u} — {v}: {NODE_NAMES.get(u,u)} ↔ {NODE_NAMES.get(v,v)}")
else: report_lines.append("No bridges found.")
bd,bp = dijkstra(ADJ,"P")
for tgt_ex in ["A","F"]:
    if bd.get(tgt_ex,float('inf'))<float('inf'):
        path_ex = reconstruct_path(bp,tgt_ex)
        report_lines.append(f"P → {tgt_ex}: {bd[tgt_ex]:.1f} min via {'->'.join(path_ex)}")
    else:
        report_lines.append(f"P → {tgt_ex}: no path found")
report_txt="\n".join(report_lines)
st.download_button("Download Project Report (txt)", data=report_txt.encode("utf-8"),
                   file_name="project_report.txt", mime="text/plain")

st.markdown("---")
st.caption("Notes: For interactive click-to-add inside Streamlit install streamlit-folium. For Google Maps live travel times supply an API key. GTFS parser uses stops.txt/stop_times.txt to add sequential stop edges.")

