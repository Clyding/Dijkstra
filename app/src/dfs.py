def dfs_find_path(graph, start, target, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]     
    visited.add(start)

    if start == target:
        return path

    for neighbor in graph[start]:
        if neighbor not in visited:
            new_path = dfs_find_path(graph, neighbor, target, path, visited)

            if new_path is not None:
                return new_path
    return None


def dfs_all_paths(graph, start, target, path=None, visited=None, all_paths=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()
    if all_paths is None:
        all_paths = []

    path = path + [start]
    visited.add(start)

    if start == target:
        all_paths.append(path)
    else:
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs_all_paths(graph, neighbor, target, path, visited.copy(), all_paths)

    return all_paths