"""
Point distribution for PEBSI

Holds the computational geometry that breaks a glacier
outline into model points and assigns each point the
share of glacier area it represents.

Three distributions are available:

'grid'      an axis-aligned lattice clipped to the outline,
            weighted by clipped Voronoi cell area
'mesh'      a patch-conforming triangular mesh: the outline
            is discretized first, the interior is filled with
            an equilateral lattice, the two are triangulated
            together and smoothed, and each point is a triangle
            centroid weighted by its Voronoi cell area
'adaptive'  the patch-conforming mesh, at whatever spacing
            each glacier's own sampling error asks for

The 'mesh' distribution is controlled by a target element
edge length in meters rather than a point count, so the same
setting transfers between glaciers of different size.

'adaptive' starts from a tolerance on glacier-wide mass balance
instead, and works back to the point count that holds the error
inside it. A fixed edge length does not do this: the error comes
from sampling a spatially varying field at finitely many points,
so it is set by how much mass balance varies across the glacier,
not by resolution alone. 
"""
# Internal libraries
import os
# External libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import shapely.geometry as geom
from shapely.ops import voronoi_diagram
from shapely import STRtree
from scipy.spatial import Delaunay
from pyproj import CRS


def get_metric_crs(gdf):
    """
    Derives a region-appropriate equal-area projection
    using the glacier centroid.

    Parameters
    ==========
    gdf : gpd.GeoDataFrame
        RGI dataframe clipped to the glacier(s) of interest
    """
    centroid = gdf.to_crs(epsg=4326).union_all().centroid
    return CRS(
        f"+proj=laea +lat_0={centroid.y:.2f} +lon_0={centroid.x:.2f} "
        f"+datum=WGS84 +units=m +no_defs"
    )


def glacier_polygon(rgi_gdf, gid):
    """
    Pulls one glacier's outline out of the RGI geodataframe
    and reprojects it into a metric CRS.

    Parameters
    ==========
    rgi_gdf : gpd.GeoDataFrame
        RGI outlines for the region
    gid : str
        Glacier ID without the 'RGI60-' prefix

    Returns
    =======
    polygon : shapely geometry
        Glacier outline in metric coordinates
    metric_crs : CRS
        The projection the polygon is in
    """
    current_glacier = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-' + gid]
    metric_crs = get_metric_crs(current_glacier)
    polygon = current_glacier.to_crs(metric_crs).unary_union
    return polygon, metric_crs


def to_latlon(xs, ys, crs):
    """
    Converts metric coordinates back to latitude and longitude.

    Parameters
    ==========
    xs, ys : 1D arrays
        Point coordinates in the metric CRS
    crs : CRS
        The projection the points are in

    Returns
    =======
    lons, lats : lists
        Point coordinates in EPSG:4326
    """
    points = [geom.Point(x, y) for x, y in zip(xs, ys)]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    points_latlon = points_gdf.to_crs(epsg=4326)
    return points_latlon.geometry.x.tolist(), points_latlon.geometry.y.tolist()


def point_budget(rgi_df, rgi_ids, n_points):
    """
    Splits a total point count between glaciers in proportion
    to their area, so bigger glaciers get proportionally more
    points. Rounding is settled by handing the leftover points
    to the glaciers with the largest rounding residuals, so the
    counts sum to exactly n_points.

    Parameters
    ==========
    rgi_df : pd.DataFrame
        RGI attributes for the region
    rgi_ids : list
        Glacier IDs without the 'RGI60-' prefix
    n_points : int
        Total points to distribute

    Returns
    =======
    budget : dict
        Glacier ID mapped to its point count
    """
    unique_ids = np.unique(rgi_ids)
    ids_fmtd = ['RGI60-' + id for id in unique_ids]
    rgi_df = rgi_df.loc[rgi_df['RGIId'].isin(ids_fmtd)]

    total_area = rgi_df['Area'].sum()
    rgi_df['exact_points'] = (rgi_df['Area'] / total_area) * n_points
    rgi_df['points'] = rgi_df['exact_points'].round().astype(int)

    # fix discrepancy to get exactly n_points
    current_sum = rgi_df['points'].sum()
    remainder = int(n_points - current_sum)

    if remainder != 0:
        # find the indices of the largest rounding fractions to adjust
        rgi_df['residual'] = rgi_df['exact_points'] - rgi_df['points']

        if remainder > 0:
            # need more points: add them to the ones that were rounded down the most
            idx = rgi_df['residual'].nlargest(remainder).index
            rgi_df.loc[idx, 'points'] += 1
        elif remainder < 0:
            # have too many points: subtract from the ones that were rounded up the most
            idx = rgi_df['residual'].nsmallest(abs(remainder)).index
            rgi_df.loc[idx, 'points'] -= 1

    budget = {}
    for gid in unique_ids:
        budget[gid] = rgi_df.loc[rgi_df['RGIId'] == 'RGI60-' + gid, 'points'].item()
    return budget


def voronoi_cells(points, polygon):
    """
    Builds each point's Voronoi cell, clipped to the polygon.

    Parameters
    ==========
    points : list of shapely Points
        Points to build cells around, in metric coordinates
    polygon : shapely geometry
        Glacier outline in the same CRS

    Returns
    =======
    cells : array of shapely geometries
        One clipped cell per point, in the order given
    """
    polygon = polygon.buffer(0)

    # gridded lattice will cause numeric issues so jitter it
    scale = np.sqrt(polygon.area / len(points)) * 1e-7
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-scale, scale, size=(len(points), 2))
    jittered = [geom.Point(p.x + dx, p.y + dy) for p, (dx, dy) in zip(points, jitter)]

    raw = np.array(voronoi_diagram(geom.MultiPoint(jittered),
                                   envelope=polygon).geoms)

    # match every cell to the one point inside it, then keep only the
    # part of the cell that is actually glacier
    point_idx, cell_idx = STRtree(raw).query(jittered, predicate='within')
    assert len(point_idx) == len(points) and len(np.unique(point_idx)) == len(points), \
        f'matched {len(np.unique(point_idx))} points to Voronoi cells, expected {len(points)}'

    cells = np.empty(len(points), dtype=object)
    cells[point_idx] = shapely.intersection(shapely.buffer(raw[cell_idx], 0), polygon)
    return cells


def voronoi_weights(points, polygon):
    """
    Weights each point by its Voronoi cell area within
    the polygon, normalized to sum to 1.

    Parameters
    ==========
    points : list of shapely Points
        Points to weight, in metric coordinates
    polygon : shapely geometry
        Glacier outline in the same CRS
    """
    weights = shapely.area(voronoi_cells(points, polygon))
    return weights / weights.sum()


def grid_polygon(polygon, target_n, crs, tolerance=0.05):
    """
    Fills a polygon with approximately target_n evenly spaced
    points using an adaptive grid spacing search. Returns lon,
    lat, and area-weight lists (weight = each point's Voronoi
    cell, clipped to the polygon and normalized to sum to 1 --
    see voronoi_weights above).

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in metric coordinates
    target_n : int
        Number of points to aim for
    crs : CRS
        The projection the polygon is in
    tolerance : float
        Acceptable deviation between actual points
        generated and target_n
    """
    xmin, ymin, xmax, ymax = polygon.bounds
    area = polygon.area

    # initial analytical guess for even grid spacing: sqrt(Area / N)
    spacing = np.sqrt(area / target_n)

    # optimization loop to fine-tune spacing to hit your exact target N count
    for _ in range(15):
        x_coords = np.arange(xmin, xmax, spacing)
        y_coords = np.arange(ymin, ymax, spacing)

        # create coordinate meshgrid matrix
        xv, yv = np.meshgrid(x_coords, y_coords)
        candidate_points = [geom.Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]

        # vectorized boundary clipping mask: Keep points strictly inside the polygon
        points_inside = [p for p in candidate_points if polygon.contains(p)]
        current_count = len(points_inside)

        # check if we are within acceptable tolerance of our target N count
        if abs(current_count - target_n) / target_n <= tolerance:
            break

        # adjust grid step density dynamically based on overshoot/undershoot
        spacing *= np.sqrt(current_count / target_n)

    weights = voronoi_weights(points_inside, polygon)

    points_gdf = gpd.GeoDataFrame(geometry=points_inside, crs=crs)
    points_latlon = points_gdf.to_crs(epsg=4326)
    return points_latlon.geometry.x.tolist(), points_latlon.geometry.y.tolist(), weights.tolist()


def distribute_grid(rgi_df, rgi_gdf, rgi_ids, n_points, tolerance=0.05):
    """
    Samples approximately n_points, evenly distributed
    inside a polygon shapefile using a grid spacing search.
    Glaciers are naturally weighted by their area
    (bigger = proportionally more points).

    Parameters
    ==========
    rgi_df : pd.DataFrame
        RGI attributes for the region
    rgi_gdf : gpd.GeoDataFrame
        RGI outlines for the region
    rgi_ids : list
        Glacier IDs without the 'RGI60-' prefix
    n_points : int
        Total points to distribute across all glaciers
    tolerance : float
        Acceptable deviation between actual points
        generated and n_points

    Returns
    =======
    lats, lons, glaciers, weights : lists
        Per-point latitude, longitude, glacier ID, and area
        weight (normalized to sum to 1 within each glacier)
    """
    unique_ids = np.unique(rgi_ids)
    budget = point_budget(rgi_df, rgi_ids, n_points)

    lats, lons, glaciers, weights = [], [], [], []
    for gid in unique_ids:
        target_n = budget[gid]
        polygon, metric_crs = glacier_polygon(rgi_gdf, gid)
        xs, ys, ws = grid_polygon(polygon, target_n, metric_crs, tolerance)

        # append lats and lons to the global list
        for lon, lat, w in zip(xs, ys, ws):
            lons.append(lon)
            lats.append(lat)
            glaciers.append(gid)
            weights.append(w)

    return lats, lons, glaciers, weights


# <<<<<< Patch-conforming triangular mesh >>>>>>

def _as_polygons(geometry):
    """
    Flattens a geometry into a list of non-empty Polygons.
    Negative buffers routinely split a glacier into several
    pieces or erase it entirely, so every stage below has to
    cope with both.
    """
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [g for g in geometry.geoms
            if g.geom_type == 'Polygon' and not g.is_empty]


def _dedupe(points, radius):
    """
    Collapses near-coincident nodes by snapping to a grid of
    cell size radius and keeping the first node in each cell.
    RGI outlines carry vertices far finer than any element
    size, and Delaunay chokes on coincident points.

    Parameters
    ==========
    points : (N, 2) array
        Candidate node coordinates
    radius : float
        Separation below which two nodes are considered one
    """
    if len(points) == 0:
        return points
    cells = np.floor(points / radius).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(first)]


def _ring_nodes(ring, h, corner_deg):
    """
    Places boundary nodes along one ring: every sharp corner is
    kept, and the rest of the ring is resampled at uniform arc
    length h.

    Resampling rather than densifying is what ties boundary
    spacing to the element size. An RGI outline carries vertices
    far finer than any element size, and keeping all of them
    would ring the margin with small elements while the interior
    stayed coarse, pulling the points toward the edge. Corners
    are kept because they carry the shape: dropping them rounds
    off the terminus and shaves real area.

    Parameters
    ==========
    ring : shapely LinearRing
        One ring of the outline
    h : float
        Target element edge length [m]
    corner_deg : float
        Turn angle above which a vertex is treated as a corner

    Returns
    =======
    nodes : (N, 2) array
        Corner nodes first, so that deduplication keeps them
        in preference to the resampled ones
    """
    # corners are judged on a ring stripped of digitization noise:
    # a sharp turn between two 20 m vertices of an RGI outline is
    # not a feature of the glacier, and treating it as one crowds
    # the margin with nodes
    smoothed = ring.simplify(h / 10)
    if smoothed.is_empty or len(np.asarray(smoothed.coords)) < 4:
        smoothed = ring

    # at least three nodes, so a nunatak smaller than one element
    # still encloses an area instead of collapsing to a line
    n = max(int(np.ceil(smoothed.length / h)), 3)
    distances = np.linspace(0, smoothed.length, n, endpoint=False)
    resampled = shapely.get_coordinates(
        shapely.line_interpolate_point(smoothed, distances))

    coords = np.asarray(smoothed.coords)[:-1]
    if len(coords) < 3:
        return resampled

    incoming = coords - np.roll(coords, 1, axis=0)
    outgoing = np.roll(coords, -1, axis=0) - coords
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    turn = np.abs(np.arctan2(cross, dot))
    corners = coords[turn > np.deg2rad(corner_deg)]

    return np.vstack([corners, resampled])


def _boundary_nodes(polygon, h, corner_deg=60.0):
    """
    Walks every ring of the geometry and returns the boundary
    nodes, which stay fixed through smoothing. Interior rings
    are included, so nunataks get a conforming boundary the
    same way the outer margin does.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in metric coordinates
    h : float
        Target element edge length [m]
    corner_deg : float
        Turn angle above which a vertex is treated as a corner
    """
    rings = []
    for poly in _as_polygons(polygon):
        for ring in [poly.exterior, *poly.interiors]:
            rings.append(_ring_nodes(ring, h, corner_deg))
    if not rings:
        return np.empty((0, 2))

    # corners come first in each ring, so they survive in
    # preference to a resampled node landing beside them
    return _dedupe(np.vstack(rings), h / 4)


def _hex_lattice(polygon, bounds, h):
    """
    Builds an equilateral triangular lattice covering bounds:
    rows are spaced h*sqrt(3)/2 apart and every other row is
    offset by h/2, which makes the natural node set for a
    triangular mesh and carries no axis preference.

    The lattice is anchored on the polygon centroid rather
    than a bounding box corner, so its phase is a property of
    the glacier rather than of the bounding box.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in metric coordinates
    bounds : tuple
        (xmin, ymin, xmax, ymax) to cover
    h : float
        Target element edge length [m]
    """
    xmin, ymin, xmax, ymax = bounds
    dy = h * np.sqrt(3) / 2
    cx, cy = polygon.centroid.x, polygon.centroid.y

    rows_lo = int(np.floor((ymin - cy) / dy))
    rows_hi = int(np.ceil((ymax - cy) / dy))

    xs, ys = [], []
    for k in range(rows_lo, rows_hi + 1):
        y = cy + k * dy
        offset = 0.0 if k % 2 == 0 else h / 2
        cols_lo = int(np.floor((xmin - cx - offset) / h))
        cols_hi = int(np.ceil((xmax - cx - offset) / h))
        row_x = cx + offset + h * np.arange(cols_lo, cols_hi + 1)
        xs.append(row_x)
        ys.append(np.full(row_x.shape, y))

    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def _triangulate(nodes, polygon):
    """
    Delaunay triangulates the nodes and keeps only the elements
    whose centroid lies inside the polygon. Delaunay always
    triangulates the convex hull, so this cull is what recovers
    conformance: it removes elements spanning a concavity
    between two tongues and elements bridging a nunatak.

    Parameters
    ==========
    nodes : (N, 2) array
        Mesh node coordinates
    polygon : shapely geometry
        Glacier outline in the same CRS

    Returns
    =======
    simplices : (M, 3) int array
        Node indices of the elements inside the glacier
    """
    if len(nodes) < 3:
        return np.empty((0, 3), dtype=int)
    try:
        tri = Delaunay(nodes)
    except Exception:
        return np.empty((0, 3), dtype=int)

    verts = nodes[tri.simplices]
    cx = verts[:, :, 0].mean(axis=1)
    cy = verts[:, :, 1].mean(axis=1)
    inside = shapely.contains_xy(polygon, cx, cy)
    return tri.simplices[inside]


def _element_areas(nodes, simplices):
    """
    Triangle areas by the shoelace formula.
    """
    v = nodes[simplices]
    cross = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
             - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    return 0.5 * np.abs(cross)


def _element_centroids(nodes, simplices):
    """
    Triangle centroids.
    """
    v = nodes[simplices]
    return v[:, :, 0].mean(axis=1), v[:, :, 1].mean(axis=1)


def _min_angles(nodes, simplices):
    """
    Smallest interior angle of each element, in degrees.
    Used only to report mesh quality.
    """
    if len(simplices) == 0:
        return np.empty(0)
    v = nodes[simplices]
    a = np.linalg.norm(v[:, 1] - v[:, 2], axis=1)
    b = np.linalg.norm(v[:, 2] - v[:, 0], axis=1)
    c = np.linalg.norm(v[:, 0] - v[:, 1], axis=1)
    eps = np.finfo(float).tiny
    angle_a = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c + eps), -1, 1))
    angle_b = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c + eps), -1, 1))
    angle_c = np.pi - angle_a - angle_b
    smallest = np.minimum(np.minimum(angle_a, angle_b), angle_c)
    return np.degrees(smallest)


def _smooth(nodes, simplices, fixed, polygon):
    """
    One round of Laplacian smoothing: every free node moves to
    the mean of its mesh neighbours. Boundary nodes are frozen,
    so the outline stays conforming, and any node the move
    would push outside the glacier stays where it was.

    Parameters
    ==========
    nodes : (N, 2) array
        Mesh node coordinates
    simplices : (M, 3) int array
        Elements defining node adjacency
    fixed : (N,) bool array
        True for boundary nodes, which do not move
    polygon : shapely geometry
        Region free nodes must stay inside
    """
    if len(simplices) == 0:
        return nodes

    edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    ends = np.concatenate([edges[:, 0], edges[:, 1]])
    others = np.concatenate([edges[:, 1], edges[:, 0]])

    total = np.zeros_like(nodes)
    np.add.at(total, ends, nodes[others])
    count = np.bincount(ends, minlength=len(nodes))

    moved = nodes.copy()
    free = (~fixed) & (count > 0)
    moved[free] = total[free] / count[free, None]

    stayed_in = shapely.contains_xy(polygon, moved[:, 0], moved[:, 1])
    moved[~stayed_in] = nodes[~stayed_in]
    return moved


def _single_point(polygon):
    """
    Fallback for a glacier too small to mesh at the requested
    edge length: one point carrying the whole area.
    """
    point = polygon.representative_point()
    diagnostics = dict(n_points=1, spacing=np.nan, mesh_area=polygon.area,
                       polygon_area=polygon.area, area_ratio=1.0,
                       min_angle=np.nan, mean_min_angle=np.nan)
    return (np.array([point.x]), np.array([point.y]),
            np.array([1.0]), diagnostics)


def defeature(polygon, h, min_feature_frac=1 / 9):
    """
    Drops interior rings far below the element size.

    A nunatak much smaller than one element cannot be meshed
    without forcing local refinement around it, which crowds
    points where the geometry is fiddly and breaks the idea
    that h is one glacier-wide resolution. Ringing the outline
    at that scale also pulls points toward the margin.

    The threshold is deliberately well below h so only features
    the mesh could never resolve are dropped: at the default
    a ring has to be smaller than an (h/3) square to go, which
    keeps the ice area added below about a percent. Dropped
    rings reappear as the mesh is refined.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in a metric CRS
    h : float
        Target element edge length [m]
    min_feature_frac : float
        Smallest interior ring kept, as a fraction of h^2

    Returns
    =======
    polygon : shapely geometry
        The outline with unresolvable interior rings filled in
    """
    threshold = min_feature_frac * h * h
    parts = []
    for poly in _as_polygons(polygon):
        keep = [ring for ring in poly.interiors
                if geom.Polygon(ring).area >= threshold]
        parts.append(geom.Polygon(poly.exterior, keep))
    if not parts:
        return polygon
    return parts[0] if len(parts) == 1 else geom.MultiPolygon(parts)


def mesh_nodes(polygon, h, n_smooth=5, seed_frac=0.65, sliver_frac=1e-3,
               min_feature_frac=1 / 9):
    """
    Builds the patch-conforming triangulation of a polygon and
    returns its raw nodes and elements, for callers that need
    the mesh itself rather than one point per element.

    The outline is discretized first so the margin is honored,
    the interior is seeded on an equilateral lattice, the two
    node sets are triangulated together, and the free nodes
    are relaxed toward equilateral with the boundary frozen.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in a metric CRS
    h : float
        Target element edge length [m]
    n_smooth : int
        Rounds of Laplacian smoothing
    seed_frac : float
        Interior nodes are seeded seed_frac*h inside the
        margin, so the first ring of elements is not crushed
        against the boundary nodes
    sliver_frac : float
        Elements smaller than sliver_frac*h^2 are discarded

    Returns
    =======
    nodes : (N, 2) array
        Mesh node coordinates
    simplices : (M, 3) int array
        Node indices of the elements covering the glacier
    """
    polygon = defeature(polygon.buffer(0), h, min_feature_frac)
    empty = (np.empty((0, 2)), np.empty((0, 3), dtype=int))
    if not _as_polygons(polygon):
        return empty

    boundary = _boundary_nodes(polygon, h)

    interior_region = polygon.buffer(-seed_frac * h)
    lattice_x, lattice_y = _hex_lattice(polygon, polygon.bounds, h)
    if len(lattice_x) and _as_polygons(interior_region):
        keep = shapely.contains_xy(interior_region, lattice_x, lattice_y)
        interior = np.column_stack([lattice_x[keep], lattice_y[keep]])
    else:
        interior = np.empty((0, 2))

    nodes = np.vstack([boundary, interior])
    if len(nodes) < 3:
        return empty

    fixed = np.zeros(len(nodes), dtype=bool)
    fixed[:len(boundary)] = True

    simplices = _triangulate(nodes, polygon)
    for _ in range(n_smooth):
        if len(simplices) == 0:
            break
        nodes = _smooth(nodes, simplices, fixed, polygon)
        simplices = _triangulate(nodes, polygon)

    if len(simplices) == 0:
        return empty

    # discard slivers only on area: culling on angle too would
    # throw away the long thin elements that fill a narrow
    # tongue, and with them real glacier area
    areas = _element_areas(nodes, simplices)
    return nodes, simplices[areas > sliver_frac * h * h]


def mesh_polygon(polygon, h, n_smooth=5, seed_frac=0.65, sliver_frac=1e-3,
                 min_feature_frac=1 / 9):
    """
    Meshes a polygon with patch-conforming triangular elements
    of target edge length h and returns one point per element.

    The outline is discretized first so the margin is honored
    and sampled, the interior is seeded on an equilateral
    lattice, the two node sets are triangulated together and
    relaxed toward equilateral, and each element contributes
    its centroid as a point. Each point is then weighted by
    its Voronoi cell clipped to the true outline.

    Nodes sit on the true outline, so the elements tile the
    whole glacier and the weights partition its area. The
    points are element centroids rather than nodes, which
    keeps every one of them well inside the ice even where an
    element touches the margin, so a point never samples an
    off-glacier DEM pixel.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in a metric CRS
    h : float
        Target element edge length [m]
    n_smooth : int
        Rounds of Laplacian smoothing
    seed_frac : float
        Interior nodes are seeded seed_frac*h inside the
        margin, so the first ring of elements is not crushed
        against the boundary nodes
    sliver_frac : float
        Elements smaller than sliver_frac*h^2 are discarded
    min_feature_frac : float
        Smallest nunatak kept, as a fraction of h^2

    Returns
    =======
    xs, ys : 1D arrays
        Point coordinates in the polygon's CRS
    weights : 1D array
        Fraction of glacier area each point represents
    diagnostics : dict
        Point count, achieved spacing, meshed area against
        polygon area, and element angle statistics
    """
    polygon = polygon.buffer(0)
    meshed = defeature(polygon, h, min_feature_frac)
    nodes, simplices = mesh_nodes(polygon, h, n_smooth=n_smooth,
                                  seed_frac=seed_frac, sliver_frac=sliver_frac,
                                  min_feature_frac=min_feature_frac)
    if len(simplices) == 0:
        return _single_point(polygon)

    areas = _element_areas(nodes, simplices)
    xs, ys = _element_centroids(nodes, simplices)

    # a filled-in nunatak is ice to the mesh, so an element centroid can
    # land on rock. Drop those: they would be simulated on bedrock slope
    # and elevation and then credited with the ice area around them.
    on_ice = shapely.contains_xy(polygon, xs, ys)
    if not on_ice.any():
        return _single_point(polygon)
    xs, ys, areas = xs[on_ice], ys[on_ice], areas[on_ice]

    # the mesh decides where the points go; the weights are read back
    # off the true outline rather than off the elements. Element areas
    # would bake in every bit of geometry the mesh had to approximate --
    # the resampled margin and the nunataks too small to resolve -- and
    # on a glacier that error dwarfs the quadrature error they would
    # otherwise save.
    weights = voronoi_weights([geom.Point(x, y) for x, y in zip(xs, ys)], polygon)

    angles = _min_angles(nodes, simplices)
    diagnostics = dict(
        n_points=len(xs),
        spacing=float(np.sqrt(4 * areas.mean() / np.sqrt(3))),
        mesh_area=float(areas.sum()),
        max_weight_ratio=float(weights.max() * len(weights)),
        polygon_area=float(polygon.area),
        area_ratio=float(areas.sum() / meshed.area),
        defeatured_ratio=float(meshed.area / polygon.area),
        min_angle=float(angles.min()),
        mean_min_angle=float(angles.mean()),
    )
    return xs, ys, weights, diagnostics


def distribute_mesh(rgi_df, rgi_gdf, rgi_ids, spacing, **kwargs):
    """
    Breaks each glacier into patch-conforming triangular
    elements of the given edge length and returns one point
    per element.

    Spacing is a length in meters, so the same setting means
    the same spatial resolution on every glacier and transfers
    between them without being refit to area. The point count
    follows from the geometry rather than being prescribed.

    Parameters
    ==========
    rgi_df : pd.DataFrame
        RGI attributes for the region
    rgi_gdf : gpd.GeoDataFrame
        RGI outlines for the region
    rgi_ids : list
        Glacier IDs without the 'RGI60-' prefix
    spacing : float
        Target element edge length [m]
    **kwargs
        Passed through to mesh_polygon

    Returns
    =======
    lats, lons, glaciers, weights : lists
        Per-point latitude, longitude, glacier ID, and area
        weight (normalized to sum to 1 within each glacier)
    """
    lats, lons, glaciers, weights = [], [], [], []
    for gid in np.unique(rgi_ids):
        polygon, metric_crs = glacier_polygon(rgi_gdf, gid)
        xs, ys, ws, _ = mesh_polygon(polygon, spacing, **kwargs)

        point_lons, point_lats = to_latlon(xs, ys, metric_crs)
        for lon, lat, w in zip(point_lons, point_lats, ws):
            lons.append(lon)
            lats.append(lat)
            glaciers.append(gid)
            weights.append(float(w))

    return lats, lons, glaciers, weights


# <<<<<< Point count from a mass balance error tolerance >>>>>>

def point_count_for_sigma(sigma, tolerance, coefficient, confidence=1.96):
    """
    Points needed to hold the error on a glacier's area-weighted mass
    balance inside a tolerance:

        N = (confidence * coefficient * sigma / tolerance) ** 2

    The error of the glacier-wide mean is a random variable of size
    coefficient * sigma / sqrt(N), so the point count scales with the
    square of both sigma and the reciprocal of the tolerance. Halving
    the tolerance costs four times the points.

    Parameters
    ==========
    sigma : float
        Standard deviation of annual mass balance across the glacier's
        points [cm w.e. a-1]
    tolerance : float
        Allowed error on the glacier-wide mean [cm w.e. a-1]
    coefficient : float
        Fitted constant relating sigma to the error of the mean
    confidence : float
        Standard normal multiplier for the confidence wanted. Defaults
        to 1.96, the project's standing choice of 95% (see
        convergence_summary.md) -- there to be read, not swept per run.

    Returns
    =======
    n_points : int
    """
    assert sigma > 0, f'sigma must be positive, got {sigma!r}'
    assert tolerance > 0, f'tolerance must be positive, got {tolerance!r}'
    n_points = (confidence * coefficient * sigma / tolerance) ** 2
    return max(int(round(n_points)), 1)


def spacing_for_target_n(polygon, target_n, tolerance=0.05, max_iter=25, **kwargs):
    """
    Finds the element edge length that meshes this outline into about
    target_n points.

    Point count falls as edge length grows, but not as a fixed multiple
    of area over edge length squared: a coarse mesh spends a larger
    share of its points resolving the outline, so that ratio drifts by
    several times across a sweep. The spacing is therefore bracketed
    and bisected on the real mesher instead of taken from a closed form.

    Parameters
    ==========
    polygon : shapely geometry
        Glacier outline in a metric CRS
    target_n : int
        Point count to aim for
    tolerance : float
        Acceptable fractional deviation from target_n
    max_iter : int
        Cap on mesh evaluations, per stage
    **kwargs
        Passed through to mesh_polygon

    Returns
    =======
    spacing : float
        Element edge length [m]
    n_points : int
        Points that spacing actually produced
    """
    assert target_n >= 1, f'target_n must be at least 1, got {target_n!r}'

    def count(h):
        return len(mesh_polygon(polygon, h, **kwargs)[0])

    def close_enough(n):
        return abs(n - target_n) <= tolerance * target_n

    # an equilateral mesh puts roughly 2.3 elements in each h^2 of area
    spacing = np.sqrt(2.3 * polygon.area / target_n)
    n_points = count(spacing)

    # straddle the target: 'fine' overshoots it, 'coarse' undershoots
    fine = (spacing, n_points) if n_points > target_n else None
    coarse = None if n_points > target_n else (spacing, n_points)

    for _ in range(max_iter):
        if close_enough(n_points) or (fine and coarse):
            break
        spacing = spacing / 1.6 if coarse else spacing * 1.6
        n_points = count(spacing)
        if n_points > target_n:
            fine = (spacing, n_points)
        else:
            coarse = (spacing, n_points)

    if close_enough(n_points):
        return spacing, n_points
    assert fine and coarse, \
        f'could not bracket {target_n} points on this outline'

    for _ in range(max_iter):
        spacing = np.sqrt(fine[0] * coarse[0])
        n_points = count(spacing)
        if close_enough(n_points):
            break
        if n_points > target_n:
            fine = (spacing, n_points)
        else:
            coarse = (spacing, n_points)

    return spacing, n_points


def load_sigma_table(fn):
    """
    Reads the per-glacier sigma table that 'adaptive' distributes from.

    Columns: rgiid, sigma_max, and optionally tolerance_cm and
    point_spacing recording a spacing already solved at that tolerance,
    so a run does not have to search for it again.

    Parameters
    ==========
    fn : str
        Path to the table

    Returns
    =======
    table : pd.DataFrame
        Indexed by RGI ID without the 'RGI60-' prefix
    """
    assert os.path.exists(fn), (
        f'No sigma table at {fn}. method_distribute=\'adaptive\' sets each '
        'glacier\'s point count from how much its mass balance varies, which '
        'has to be measured first -- run point_density_rule.py.')

    table = pd.read_csv(fn, dtype={'rgiid': str})
    missing = {'rgiid', 'sigma_max'} - set(table.columns)
    assert not missing, f'{fn} is missing columns {sorted(missing)}'
    return table.set_index('rgiid')


def distribute_rule(rgi_df, rgi_gdf, rgi_ids, table_fn, tolerance,
                    coefficient, confidence=1.96, **kwargs):
    """
    Meshes each glacier at the resolution its own mass balance variance
    asks for, rather than at a shared edge length or a point count fit
    to area.

    A glacier needs more points when its mass balance varies more
    across its own surface, which is not what area predicts: the
    largest glacier in a set is not reliably the one that needs the
    most points. sigma_max in the table is the largest sigma measured
    anywhere in the parameter range being calibrated over, so one mesh
    holds the tolerance everywhere in that range.

    Parameters
    ==========
    rgi_df : pd.DataFrame
        RGI attributes for the region
    rgi_gdf : gpd.GeoDataFrame
        RGI outlines for the region
    rgi_ids : list
        Glacier IDs without the 'RGI60-' prefix
    table_fn : str
        Path to the sigma table
    tolerance : float
        Allowed error on each glacier-wide mean [cm w.e. a-1]
    coefficient : float
        Fitted constant relating sigma to the error of the mean
    confidence : float
        Standard normal multiplier for the confidence wanted
    **kwargs
        Passed through to mesh_polygon

    Returns
    =======
    lats, lons, glaciers, weights : lists
        Per-point latitude, longitude, glacier ID, and area
        weight (normalized to sum to 1 within each glacier)
    """
    table = load_sigma_table(table_fn)

    lats, lons, glaciers, weights = [], [], [], []
    for gid in np.unique(rgi_ids):
        assert gid in table.index, (
            f'{gid} is not in {table_fn}. Its point count depends on how '
            'much its mass balance varies across the glacier, which has to '
            'be measured -- run point_density_rule.py for it first.')

        row = table.loc[gid]
        target_n = point_count_for_sigma(
            float(row['sigma_max']), tolerance, coefficient, confidence)
        polygon, metric_crs = glacier_polygon(rgi_gdf, gid)

        # reuse a spacing already solved at this tolerance; searching for
        # one costs a dozen meshes of a glacier that may be large
        solved = np.nan
        if {'tolerance_cm', 'point_spacing'} <= set(table.columns):
            if np.isclose(float(row['tolerance_cm']), tolerance):
                solved = float(row['point_spacing'])
        spacing = solved if np.isfinite(solved) else \
            spacing_for_target_n(polygon, target_n, **kwargs)[0]

        xs, ys, ws, _ = mesh_polygon(polygon, spacing, **kwargs)
        point_lons, point_lats = to_latlon(xs, ys, metric_crs)
        for lon, lat, w in zip(point_lons, point_lats, ws):
            lons.append(lon)
            lats.append(lat)
            glaciers.append(gid)
            weights.append(float(w))

    return lats, lons, glaciers, weights
