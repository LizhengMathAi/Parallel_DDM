import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from static.algorithms.domain_decomposition_methods import utils


class Canvas:
    """
    * Since the color of simplex rely on the position of center, all coordinate should be in range [0, 1].
    * If you have not demand of show this image, just want to do the DDM task, no range limit here.
    """
    static_points = None
    dynamic_points = None
    eps = None
    infimum = None

    points = None
    simplices = None

    fig = None
    ax = None
    scale = 0

    def draw(self):
        pass

    def key_press_event(self, event):
        if event.key == 'up':
            self.scale = min(self.scale + 0.4, 2)
        elif event.key == 'down':
            self.scale = max(self.scale - 0.4, 0)
        elif event.key == 'u':
            for _ in range(10):
                self.static_points = utils.update(self.dynamic_points, self.static_points, self.eps)
            self.points = np.vstack([self.dynamic_points, self.static_points])
            self.simplices = Delaunay(self.points).simplices
        elif event.key == 'm':
            for _ in range(10):
                self.static_points = utils.merge(self.dynamic_points, self.static_points, self.infimum)
            self.points = np.vstack([self.dynamic_points, self.static_points])
            self.simplices = Delaunay(self.points).simplices
        elif event.key == 't':
            self.points = utils.train(self.dynamic_points, self.static_points)
            self.simplices = Delaunay(self.points).simplices
        elif event.key == 'r':
            self.points = utils.refine_mesh(self.points)
            self.simplices = Delaunay(self.points).simplices
        else:
            return

        # remove invalid simplices
        volumes = np.linalg.det(self.points[self.simplices[:, :-1], :] - self.points[self.simplices[:, [-1]], :])
        valid_indices = [i for i, v in enumerate(volumes) if v != 0]
        self.simplices = self.simplices[valid_indices]

        self.ax.clear()
        self.draw()
        self.fig.canvas.draw()


class Canvas2D(Canvas):
    def __init__(self, boundary_points, inner_points, eps=0.01, infimum=0.1):
        self.dynamic_points = boundary_points
        self.static_points = inner_points
        self.eps = eps
        self.infimum = infimum

        self.points = np.vstack([boundary_points, inner_points])
        self.simplices = Delaunay(self.points).simplices

        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111)

        self.draw()

        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)

    def draw(self):
        common_center = np.mean(self.points, axis=0)

        polygons, colors = [], []
        for simplex in self.simplices:
            individual_center = np.mean(self.points[simplex], axis=0)
            offset = self.scale * np.expand_dims(individual_center - common_center, axis=0)
            polygons.append(self.points[simplex] + offset)
            color = np.array(individual_center / np.linalg.norm(individual_center) * 1000, dtype=np.int) / 1000
            colors.append([color[0], 0, color[1]])

        polygons = np.array(polygons)
        colors = np.array(colors)
        self.ax.add_collection(PolyCollection(polygons, color=colors, alpha=0.5))

        x_min = np.min(polygons[:, :, 0])
        x_max = np.max(polygons[:, :, 0])
        y_min = np.min(polygons[:, :, 1])
        y_max = np.max(polygons[:, :, 1])
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    @classmethod
    def demo(cls, dir_path, n_inner):
        boundary_points = np.array([[np.cos(r) * 0.5 + 0.5, np.sin(r) * 0.5 + 0.5] for r in np.linspace(0, 2 * np.pi, 10)])
        inner_points = np.array([pt for pt in np.random.rand(1000, 2) if np.linalg.norm(pt - np.array([0.5, 0.5])) < 0.5])[:n_inner]

        origin = cls(boundary_points=boundary_points, inner_points=inner_points)
        origin.fig.savefig(dir_path + "/origin2d.png")

        target = cls(boundary_points=boundary_points, inner_points=inner_points)
        target.points = utils.train(target.dynamic_points, target.static_points)
        target.simplices = Delaunay(target.points).simplices
        target.ax.clear()
        target.draw()
        target.fig.canvas.draw()
        target.fig.savefig(dir_path + "/target2d.png")


class Canvas3D(Canvas):
    def __init__(self, boundary_points, inner_points, eps=0.01, infimum=0.1):
        self.dynamic_points = boundary_points
        self.static_points = inner_points
        self.eps = eps
        self.infimum = infimum

        self.points = np.vstack([boundary_points, inner_points])
        self.simplices = Delaunay(self.points).simplices

        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.gca(projection='3d')

        self.draw()

        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)

    def draw(self):
        common_center = np.mean(self.points, axis=0)
        x_min, y_min, z_min = x_max, y_max, z_max = common_center

        # boundary_pts, inner_pts = [], []
        for simplex in self.simplices:
            individual_center = np.mean(self.points[simplex], axis=0)
            offset = self.scale * np.expand_dims(individual_center - common_center, axis=(0, 1))
            polygons = np.array([self.points[np.roll(simplex, stride)[:3]] for stride in range(4)]) + offset
            color = np.array(individual_center / np.linalg.norm(individual_center) * 1000, dtype=np.int) / 1000
            self.ax.add_collection3d(Poly3DCollection(polygons, color=color, alpha=0.5))

            x_min = min(x_min, np.min(polygons[:, :, 0]))
            x_max = max(x_max, np.max(polygons[:, :, 0]))
            y_min = min(y_min, np.min(polygons[:, :, 1]))
            y_max = max(y_max, np.max(polygons[:, :, 1]))
            z_min = min(z_min, np.min(polygons[:, :, 2]))
            z_max = max(z_max, np.max(polygons[:, :, 2]))

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)

    @classmethod
    def demo(cls, dir_path, n_inner):
        boundary_points = np.array([[i // 4, i % 4 // 2, i % 2] for i in range(8)], dtype=np.float)
        inner_points = np.random.rand(n_inner, 3)

        origin = cls(boundary_points=boundary_points, inner_points=inner_points)
        origin.scale = 2.
        origin.ax.clear()
        origin.draw()
        origin.fig.canvas.draw()
        origin.fig.savefig(dir_path + "/origin3d.png")

        target = cls(boundary_points=boundary_points, inner_points=inner_points)
        target.scale = 2.
        target.points = utils.train(target.dynamic_points, target.static_points)
        target.simplices = Delaunay(target.points).simplices
        target.ax.clear()
        target.draw()
        target.fig.canvas.draw()
        target.fig.savefig(dir_path + "/target3d.png")


if __name__ == "__main__":
    # This method is appropriate for the domain decomposition task of convex hull.
    # * Press `up` key to decompose graphics.
    # * Press `down` key to compose graphics.
    # * Press `u` key to update(optimize) the positions of inner points.
    # * Press `m` key to merge(eliminate) some extra inner points.
    # * Press `r` key to refine the mesh.
    # * Press `t` key to train(the ultimate method) the mesh.
    np.random.seed(0)
    np.set_printoptions(precision=2)

    # Initial 2D cube demo.
    cube2d = Canvas2D(
        boundary_points=np.array([[i // 2, i % 2] for i in range(4)], dtype=np.float),
        inner_points=np.random.rand(16, 2)
    )

    # Initial 2D ball demo.
    ball2d = Canvas2D(
        boundary_points=np.array([[np.cos(r)*0.5+0.5, np.sin(r)*0.5+0.5] for r in np.linspace(0, 2 * np.pi, 10)]),
        inner_points=np.array([pt for pt in np.random.rand(8, 2) if np.linalg.norm(pt - np.array([0.5, 0.5])) < 0.5])
    )

    # Initial 3D cube canvas.
    cube3d = Canvas3D(
        boundary_points=np.array([[i // 4, i % 4 // 2, i % 2] for i in range(8)], dtype=np.float),
        inner_points=np.random.rand(8, 3)
    )

    plt.show()
