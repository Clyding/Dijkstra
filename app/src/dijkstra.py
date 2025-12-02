import heapq

# City map with travel times (minutes)
NODE_NAMES = {
    "P": "Penn Station (Transit Hub)",
    "A": "National Aquarium",
    "B": "M&T Bank Stadium",
    "C": "Convention Center (Light Rail Stop)",
    "D": "Fells Point",
    "E": "Federal Hill",
    "F": "Fort McHenry",
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

def gps_shortest_path(graph, start, end):
    distances = {loc: float('inf') for loc in graph}
    distances[start] = 0  
    previous = {loc: None for loc in graph}
    pq = [(0, start)]
    while pq:
        curr_time, location = heapq.heappop(pq)
        if location == end:
            path = []
            while location:
                path.append(location)
                location = previous[location]
            return curr_time, path[::-1]
        if curr_time > distances[location]:
            continue
        for neighbor, time in graph[location].items():
            new_time = curr_time + time
            if new_time < distances[neighbor]:
                distances[neighbor] = new_time
                previous[neighbor] = location
                heapq.heappush(pq, (new_time, neighbor))
    return float('inf'), []  