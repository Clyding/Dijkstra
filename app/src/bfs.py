from collections import deque 

def bfs(graph, start):
    
    visited = set()              
    queue = deque([start])       
    visited.add(start)           
    result = []                

    while queue:
        node = queue.popleft()

        result.append(node)
        print(f"Visiting node: {node}")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)      
                queue.append(neighbor)    

        print(f"Queue state: {list(queue)}")

    return result


graph = inner_harbor_map = {
  "Inner Harbor (Central)": {
    "National Aquarium": 0.2,
    "Maryland Science Center": 0.3,
    "Federal Hill Park": 0.8,
    "Fells Point (Broadway Market)": 1.2,
    "Harbor East (National Katyn Memorial)": 0.6,
    "Little Italy (center)": 0.7,
    "Mount Vernon (Washington Monument)": 1.0,
    "Oriole Park at Camden Yards": 0.8
  },
  "National Aquarium": {
    "Inner Harbor (Central)": 0.2,
    "Harbor East (National Katyn Memorial)": 0.4
  },
  "Maryland Science Center": {
    "Inner Harbor (Central)": 0.3,
    "Federal Hill Park": 0.5,
    "American Visionary Art Museum": 0.6
  },
  "Federal Hill Park": {
    "Maryland Science Center": 0.5,
    "Inner Harbor (Central)": 0.8,
    "Baltimore Museum of Industry": 1.0,
    "Locust Point (Fort McHenry)": 2.2
  },
  "Fells Point (Broadway Market)": {
    "Inner Harbor (Central)": 1.2,
    "Harbor East (National Katyn Memorial)": 0.7,
    "Canton Waterfront Park": 1.5,
    "Little Italy (center)": 0.5
  },
  "Harbor East (National Katyn Memorial)": {
    "Inner Harbor (Central)": 0.6,
    "National Aquarium": 0.4,
    "Fells Point (Broadway Market)": 0.7,
    "Little Italy (center)": 0.4
  },
  "Little Italy (center)": {
    "Inner Harbor (Central)": 0.7,
    "Harbor East (National Katyn Memorial)": 0.4,
    "Fells Point (Broadway Market)": 0.5
  },
  "Mount Vernon (Washington Monument)": {
    "Inner Harbor (Central)": 1.0,
    "Walters Art Museum": 0.1,
    "Penn Station (Baltimore)": 0.7
  },
  "Oriole Park at Camden Yards": {
    "Inner Harbor (Central)": 0.8,
    "M&T Bank Stadium": 0.4
  },
  "American Visionary Art Museum": {
    "Maryland Science Center": 0.6,
    "Baltimore Museum of Industry": 0.4
  },
  "Baltimore Museum of Industry": {
    "American Visionary Art Museum": 0.4,
    "Federal Hill Park": 1.0,
    "Locust Point (Fort McHenry)": 1.5
  },
  "Locust Point (Fort McHenry)": {
    "Baltimore Museum of Industry": 1.5,
    "Federal Hill Park": 2.2
  },
  "Canton Waterfront Park": {
    "Fells Point (Broadway Market)": 1.5
  },
  "Walters Art Museum": {
    "Mount Vernon (Washington Monument)": 0.1
  },
  "Penn Station (Baltimore)": {
    "Mount Vernon (Washington Monument)": 0.7
  },
  "M&T Bank Stadium": {
    "Oriole Park at Camden Yards": 0.4
  }
}

result = bfs(graph, 'A')
print(f"BFS traversal: {result}")

