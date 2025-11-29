import folium
import heapq

# ============================
# 1. NODE COORDINATES
# ============================
NODE_COORDS = {
    "P": (39.3079, -76.6157),
    "A": (39.2857, -76.6081),
    "F": (39.2813, -76.5803),
    "B": (39.2839, -76.6217),
    "C": (39.2851, -76.6187),
    "E": (39.2789, -76.6120),
    "G": (39.2833, -76.6026),
    "H": (39.2784, -76.6000),
    "I": (39.2850, -76.6132),
    "J": (39.2913, -76.6052),
    "K": (39.2708, -76.5930),
    "L": (39.2867, -76.6030),
    "M": (39.3009, -76.6158),
    "O": (39.2821, -76.6188),
    "Q": (39.2911, -76.6208),
    "R": (39.2847, -76.6220),
    "S": (39.2930, -76.6140),
    "T": (39.2844, -76.6105)
}

# ============================
# 2. ADJACENCY LIST
# ============================

DEFAULT_ADJ = {
    "P": [("M", 6, "walk"), ("L", 8, "walk"), ("C", 12, "walk"), ("S",10,"walk")],
    "M": [("P", 6, "walk"), ("B", 9, "light_rail"), ("C", 7, "light_rail")],
    "L": [("P", 8, "walk"), ("Q", 10, "walk")],
    "Q": [("L", 10, "walk"), ("C", 7, "walk")],
    "C": [("A", 3, "walk"), ("O", 2, "walk"), ("B", 10, "walk"), ("D", 9, "walk"),
          ("E", 7, "walk"), ("G", 8, "walk"), ("I",5,"walk"), ("R",5,"walk"),
          ("S",8,"walk"),("T",7,"walk"),("P",12,"walk")],
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

# ============================
# 3. DIJKSTRA SHORTEST PATH
# ============================

def dijkstra(start, end):
    pq = [(0, start, [])]
    visited = set()

    while pq:
        dist, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        path = path + [node]

        if node == end:
            return path

        for neighbor, cost, mode in DEFAULT_ADJ.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (dist + cost, neighbor, path))

    return None


# Example: show shortest path C → F
SHORTEST_PATH = dijkstra("C", "F")

# ============================
# 4. CREATE MAP
# ============================

m = folium.Map(location=[39.285, -76.61], zoom_start=14)

# ============================
# 5. ADD NODES
# ============================

for node, (lat, lon) in NODE_COORDS.items():
    folium.Marker(
        location=[lat, lon],
        tooltip=f"{node}",
        popup=f"{node}",
        icon=folium.DivIcon(
            html=f"""
                <div style="
                    font-size:14px;
                    font-weight:bold;
                    color:black;
                    background:white;
                    border:1px solid black;
                    border-radius:4px;
                    padding:2px;
                ">{node}</div>
            """
        ),
    ).add_to(m)

# ============================
# 6. ADD EDGES (color by mode)
# ============================

MODE_COLORS = {
    "walk": "blue",
    "light_rail": "red",
    "water_taxi": "cyan"
}

for node, edges in DEFAULT_ADJ.items():
    lat1, lon1 = NODE_COORDS[node]
    for neighbor, cost, mode in edges:
        lat2, lon2 = NODE_COORDS[neighbor]
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=MODE_COLORS.get(mode, "gray"),
            weight=4,
            tooltip=f"{node} → {neighbor} ({cost} min, {mode})"
        ).add_to(m)

# ============================
# 7. HIGHLIGHT SHORTEST PATH
# ============================

if SHORTEST_PATH:
    path_lines = []
    for i in range(len(SHORTEST_PATH)-1):
        n1, n2 = SHORTEST_PATH[i], SHORTEST_PATH[i+1]
        path_lines.append((NODE_COORDS[n1], NODE_COORDS[n2]))

    for (lat1, lon1), (lat2, lon2) in path_lines:
        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color="yellow",
            weight=7,
            tooltip="Shortest Path"
        ).add_to(m)

# Show final map (Jupyter)
m

# For Streamlit:
# import streamlit as st
# st.components.v1.html(m._repr_html_(), height=700)
