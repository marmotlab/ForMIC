import heapq
import numpy as np


def astar2d(array, start, goal):
	def heuristic(a, b):
		return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

	neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4-connected grid, change if you wanna add diagonals

	close_set = set()
	came_from = {}
	gscore = {start: 0}
	fscore = {start: heuristic(start, goal)}
	oheap = []

	heapq.heappush(oheap, (fscore[start], start))

	while oheap:

		current = heapq.heappop(oheap)[1]

		if current == goal:
			data = []
			while current in came_from:
				data.append(current)
				current = came_from[current]
			data.append(start)
			return data[::-1]

		close_set.add(current)
		for i, j in neighbors:
			neighbor = current[0] + i, current[1] + j
			tentative_g_score = gscore[current] + heuristic(current, neighbor)
			if 0 <= neighbor[0] < array.shape[0]:
				if 0 <= neighbor[1] < array.shape[1]:
					if array[neighbor[0]][neighbor[1]] == 1:
						continue
				else:
					# array bound y walls
					continue
			else:
				# array bound x walls
				continue

			if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
				continue

			if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
				came_from[neighbor] = current
				gscore[neighbor] = tentative_g_score
				fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
				heapq.heappush(oheap, (fscore[neighbor], neighbor))

	return False

if __name__ == '__main__':
	world = np.zeros((10,10))
	print(astar2d(world, (1,1), (8,8)))
	print(astar2d(world, (1,1), (1,2)))
