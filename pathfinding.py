import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have Qt installed

import matplotlib.pyplot as plt
import numpy as np
import heapq

class DijkstraMap:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def shortest_path(self, start, end):
        heap = [(self.grid[start[0]][start[1]], start)]
        distances = {start: self.grid[start[0]][start[1]]}
        prev = {}

        while heap:
            curr_cost, (r, c) = heapq.heappop(heap)
            if (r, c) == end:
                break
            for nr, nc in self.neighbors(r, c):
                new_cost = curr_cost + self.grid[nr][nc]
                if (nr, nc) not in distances or new_cost < distances[(nr, nc)]:
                    distances[(nr, nc)] = new_cost
                    prev[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (new_cost, (nr, nc)))

        path = []
        node = end
        if node in prev or node == start:
            while node != start:
                path.append(node)
                node = prev[node]
            path.append(start)
            path.reverse()
        return distances.get(end, float('inf')), path

    def visualize(self, path):
        grid_display = np.array(self.grid)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_display, cmap='gray_r', origin='upper')

        if path:
            y_coords, x_coords = zip(*path)
            plt.plot(x_coords, y_coords, color='red', linewidth=2, marker='o')

        plt.title('Dijkstra Shortest Path Visualization')
        plt.colorbar(label='Cost')
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == "__main__":
    grid = [
        [1, 1, 1, 1, 1],
        [1, 9, 9, 9, 1],
        [1, 1, 1, 9, 1],
        [9, 9, 1, 9, 1],
        [1, 1, 1, 1, 1]
    ]

    dmap = DijkstraMap(grid)
    start = (0, 0)
    end = (4, 4)
    cost, path = dmap.shortest_path(start, end)
    print("Minimum cost:", cost)
    print("Path:", path)
    dmap.visualize(path)
