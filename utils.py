import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, csgraph


def projection(vertices, index):
    index %= vertices.__len__()
    offset_index = (index + 1) % vertices.__len__()

    sorted_indices = [i for i in range(vertices.__len__()) if i not in [index, offset_index]] + [index]
    offset_vertices = vertices[sorted_indices] - vertices[[offset_index], :]
    q, r = np.linalg.qr(offset_vertices.T)
    unsafe_projection = r[:, -1]
    unsafe_projection[-1] = 0
    return q@unsafe_projection + vertices[offset_index]


def gradients(static_points, dynamic_points, eps=1e-2):
    n = static_points.shape[-1]
    grad = np.zeros_like(dynamic_points)
    for i in range(dynamic_points.shape[0]):
        for j in range(dynamic_points.shape[1]):
            delta = np.zeros_like(dynamic_points)
            delta[i, j] += eps

            forward_points = np.vstack([static_points, dynamic_points + delta])
            edges_indices = Delaunay(forward_points).simplices[:, [[i, j] for i in range(1, n + 1) for j in range(i)]]
            edges = forward_points[np.reshape(edges_indices, (-1, 2)), :]
            forward_value = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=-1)
            forward_value = np.sort(forward_value)[::-1]

            backward_points = np.vstack([static_points, dynamic_points - delta])
            edges_indices = Delaunay(backward_points).simplices[:, [[i, j] for i in range(1, n + 1) for j in range(i)]]
            edges = backward_points[np.reshape(edges_indices, (-1, 2)), :]
            backward_value = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=-1)
            backward_value = np.sort(backward_value)[::-1]

            min_len = min(forward_value.__len__(), backward_value.__len__())
            for f, b in zip(forward_value[:min_len], backward_value[:min_len]):
                if np.sign(f - b) != 0:
                    grad[i, j] = np.sign(f - b)
                    break
    return grad


def update(static_points, dynamic_points, eps=1e-2):
    """This method will decent objective function by move all dynamic nodes."""
    stride = eps
    grads = gradients(static_points, dynamic_points, eps=stride)
    while np.max(np.abs(grads)) < 0.5 and stride > eps ** 2:  # If grad is invalid.
        stride *= 0.1
        grads = gradients(static_points, dynamic_points, eps=stride)
    dynamic_points = dynamic_points - stride * grads

    return dynamic_points


def merge(static_points, dynamic_points, infimum=1e-1):
    """This method will merge some dynamic nodes or replace some dynamic nodes with their projection."""
    # merge close points.
    points = np.vstack([static_points, dynamic_points])
    n_static, n_points = static_points.__len__(), points.__len__()
    dis = np.linalg.norm(np.expand_dims(points, axis=1) - np.expand_dims(points, axis=0), axis=-1)
    dis += np.max(dis) * np.eye(n_points)
    graph = csr_matrix(dis < infimum, dtype=np.int)
    graph_dis = csgraph.dijkstra(csgraph=graph, directed=False) + np.eye(n_points)
    cluster, invalid_indices = [], []
    for i in range(n_static, n_points):
        if i in invalid_indices:
            continue
        cluster.append([])
        for j in range(n_static, n_points):
            if graph_dis[i, j] != np.inf:
                cluster[-1].append(j)
                invalid_indices.append(j)
    dynamic_points = np.stack([np.mean(points[route, :], axis=0) for route in cluster], axis=0)

    # remove points which near to boundary.
    delaunay = Delaunay(static_points)
    for i in range(dynamic_points.__len__()):
        for surface in static_points[delaunay.convex_hull, :]:
            vertices = np.vstack([np.expand_dims(dynamic_points[i], axis=0), surface])
            projection_point = projection(vertices, index=0)
            if np.linalg.norm(dynamic_points[i] - projection_point) < infimum:
                dynamic_points[i] = projection_point
                break

    # offset point in thin simplex.
    points = np.vstack([static_points, dynamic_points])
    n_static, n_points, dim = static_points.__len__(), points.__len__(), points.shape[-1]
    delaunay, flag = Delaunay(points), False
    for simplex in delaunay.simplices:
        if flag:
            break
        for k in range(dim + 1):
            if simplex[k] >= n_static:
                projection_point = projection(points[simplex], index=k)
                if np.linalg.norm(points[simplex[k]] - projection_point) < infimum:
                    points[simplex[k]] = projection_point
                    flag = True
                    break
    dynamic_points = points[n_static:]

    return dynamic_points


def train(static_points, dynamic_points=None, eps=1e-2, infimum=1e-1, n_store=10):
    """This method will optimize all points in internal region. All survivors will be regarded as the nodes in mesh."""
    if dynamic_points is None:
        return static_points

    def loop(method, static_pts, dynamic_pts, **kwargs):
        record_collections = []
        while True:
            dynamic_pts = method(static_pts, dynamic_pts, **kwargs)
            pts = np.vstack([static_pts, dynamic_pts])
            simplices = Delaunay(pts).simplices
            centers = np.array([np.mean(pts[spx], axis=0) / eps for spx in simplices], np.int) * eps
            if record_collections:
                flag = False
                for record in record_collections:
                    try:
                        if np.sum(np.abs(record - centers)) == 0.:
                            flag = True
                            break
                    except ValueError:
                        pass
                if flag:
                    break
            record_collections.append(centers)
            if record_collections.__len__() > n_store:
                record_collections = record_collections[-n_store:]
        return dynamic_pts

    # optimize inner points twice.
    dynamic_points = loop(update, static_points, dynamic_points, eps=eps)
    dynamic_points = loop(merge, static_points, dynamic_points, infimum=infimum)
    dynamic_points = loop(update, static_points, dynamic_points, eps=eps)
    dynamic_points = loop(merge, static_points, dynamic_points, infimum=infimum)
    return np.vstack([static_points, dynamic_points])


def refine_mesh(points):
    nn, dim = points.shape
    edges = Delaunay(points).simplices[:, [[i, j] for i in range(1, dim + 1) for j in range(i)]]  # [NT, NE, 2]
    row = np.minimum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
    col = np.maximum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
    indices = np.unique(np.stack([row, col], axis=1), axis=0)
    points = np.vstack([points, np.mean(points[indices, :], axis=1)])
    return np.unique(points, axis=0)


if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=2)

    # 4-D demo
    print("4-D demo " + "=" * 64)
    boundary_scatters = np.array([[i // 8, i % 8 // 4, i % 4 // 2, i % 2] for i in range(16)], dtype=np.float)
    inner_scatters = np.random.rand(4, 4)
    train(boundary_scatters, inner_scatters, eps=1e-2, infimum=1e-1, n_store=10)

    # 3-D demo
    print("3-D demo " + "=" * 64)
    boundary_scatters = np.array([[i // 4, i % 4 // 2, i % 2] for i in range(8)], dtype=np.float)
    inner_scatters = np.random.rand(8, 3)
    train(boundary_scatters, inner_scatters, eps=1e-2, infimum=1e-1, n_store=10)

    # 2-D demo
    print("2-D demo " + "=" * 64)
    boundary_scatters = np.array([[i // 2, i % 2] for i in range(4)], dtype=np.float)
    inner_scatters = np.random.rand(16, 2)
    train(boundary_scatters, inner_scatters, eps=1e-2, infimum=1e-1, n_store=10)
