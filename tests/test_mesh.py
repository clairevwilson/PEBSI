"""
Tests for the point distribution geometry in pebsi/io/mesh.py.

These run on synthetic polygons, so they need no RGI, DEM or climate
data and finish in seconds.
"""
import numpy as np
import pytest
import shapely
import shapely.geometry as geom

from pebsi.io import mesh


@pytest.fixture(scope='module')
def concave_with_hole():
    """
    An L-shaped polygon with a hole: concave enough that a Delaunay
    triangulation of its nodes spans the notch, and holed enough to
    check that nunataks are excluded.
    """
    outer = geom.Polygon([(0, 0), (4000, 0), (4000, 1500), (1500, 1500),
                          (1500, 4000), (0, 4000)])
    hole = geom.Point(700, 700).buffer(350)
    return outer.difference(hole)


@pytest.fixture(scope='module')
def blob():
    """A convex-ish smooth polygon, the easy case."""
    return geom.Point(0, 0).buffer(2000)


def test_weights_sum_to_one(concave_with_hole):
    _, _, weights, _ = mesh.mesh_polygon(concave_with_hole, 200)
    assert np.isclose(weights.sum(), 1.0)


def test_points_are_inside_the_glacier(concave_with_hole):
    xs, ys, _, _ = mesh.mesh_polygon(concave_with_hole, 200)
    assert shapely.contains_xy(concave_with_hole, xs, ys).all()


def test_no_points_in_the_hole(concave_with_hole):
    xs, ys, _, _ = mesh.mesh_polygon(concave_with_hole, 200)
    hole = geom.Point(700, 700).buffer(350)
    assert not shapely.contains_xy(hole.buffer(-20), xs, ys).any()


def test_mesh_tiles_the_polygon(concave_with_hole):
    """
    The elements must cover the glacier: if they do not, the weights
    still sum to 1 but they are a partition of the wrong shape.
    """
    for h in (400, 200, 100):
        _, _, _, diag = mesh.mesh_polygon(concave_with_hole, h)
        assert abs(diag['area_ratio'] - 1.0) < 0.02, \
            f"h={h} meshed {diag['area_ratio']:.4f} of the polygon area"


def test_point_count_falls_monotonically_with_spacing(concave_with_hole):
    """
    The resolution knob has to behave like one: refining must never
    reduce the point count, or a convergence sweep is not interpretable.
    """
    spacings = [500, 400, 300, 200, 150, 100]
    counts = [mesh.mesh_polygon(concave_with_hole, h)[3]['n_points'] for h in spacings]
    assert counts == sorted(counts), f'counts {counts} for spacings {spacings}'


def test_linear_field_is_integrated_accurately():
    """
    The area-weighted mean of f(x,y)=x over a polygon is exactly its
    centroid x, so the weighted sum measures the weighting alone. The
    weights are Voronoi cells read off the true outline, so this holds
    to a small fraction of a percent at every usable resolution.
    """
    outer = geom.Polygon([(0, 0), (4000, 0), (4000, 1500), (1500, 1500),
                          (1500, 4000), (0, 4000)])
    exact = outer.centroid.x
    for h in (400, 200, 100):
        xs, _, weights, _ = mesh.mesh_polygon(outer, h)
        assert float((weights * xs).sum()) == pytest.approx(exact, rel=1e-3), f'h={h}'


def test_linear_field_converges_under_refinement(concave_with_hole):
    """
    Refining has to close the gap rather than leave it wandering,
    which is what makes a convergence sweep interpretable.
    """
    exact = concave_with_hole.centroid.x

    errors = []
    for h in (400, 200, 100):
        xs, _, weights, _ = mesh.mesh_polygon(concave_with_hole, h)
        errors.append(abs(float((weights * xs).sum()) - exact) / exact)

    assert errors[-1] < errors[0], f'errors {errors}'
    assert errors[-1] < 1e-4, f'finest mesh still off by {errors[-1]:.2e}'


def test_weights_stay_close_to_uniform(concave_with_hole):
    """
    A well-spaced mesh should give every point a similar share of the
    glacier. One point carrying a large multiple of the others makes
    the glacier-wide mean hostage to a single simulation.
    """
    for h in (300, 200, 100):
        _, _, _, diag = mesh.mesh_polygon(concave_with_hole, h)
        assert diag['max_weight_ratio'] < 3.0, \
            f"h={h} gave one point {diag['max_weight_ratio']:.1f}x the uniform weight"


def test_is_deterministic(concave_with_hole):
    """No RNG anywhere, so a rerun must reproduce the mesh exactly."""
    first = mesh.mesh_polygon(concave_with_hole, 250)
    second = mesh.mesh_polygon(concave_with_hole, 250)
    for a, b in zip(first[:3], second[:3]):
        assert np.array_equal(a, b)


def test_multipolygon_is_meshed_whole():
    """Two disjoint lobes: both get points, and the weights still sum to 1."""
    lobes = geom.MultiPolygon([geom.Point(0, 0).buffer(800),
                               geom.Point(4000, 0).buffer(500)])
    xs, ys, weights, diag = mesh.mesh_polygon(lobes, 150)
    assert np.isclose(weights.sum(), 1.0)
    assert (xs < 2000).any() and (xs > 2000).any()
    assert abs(diag['area_ratio'] - 1.0) < 0.02


def test_glacier_smaller_than_one_element_still_gets_a_point(blob):
    """A tiny glacier must not vanish from the simulation."""
    xs, ys, weights, _ = mesh.mesh_polygon(blob, 50_000)
    assert len(xs) == 1
    assert np.isclose(weights.sum(), 1.0)
    assert shapely.contains_xy(blob, xs, ys).all()


def test_weights_are_never_negative_or_nan(concave_with_hole):
    _, _, weights, _ = mesh.mesh_polygon(concave_with_hole, 200)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)


def test_mesh_nodes_matches_mesh_polygon(concave_with_hole):
    """The plotting entry point and the distribution share one mesh."""
    nodes, simplices = mesh.mesh_nodes(concave_with_hole, 250)
    xs, ys, _, diag = mesh.mesh_polygon(concave_with_hole, 250)
    assert len(simplices) == diag['n_points'] == len(xs)
    assert nodes[simplices][:, :, 0].mean(axis=1) == pytest.approx(xs)


def test_tiny_nunataks_are_dropped_and_return_under_refinement():
    """
    A nunatak far below the element size cannot be meshed without
    crowding points around it, so it is filled in at coarse
    resolution and has to reappear once the mesh can resolve it.
    """
    outer = geom.Point(0, 0).buffer(3000)
    tiny = geom.Point(500, 500).buffer(60)
    big = geom.Point(-1200, 0).buffer(700)
    polygon = outer.difference(tiny).difference(big)

    # h=600: the 60 m nunatak is far below h/3, the 700 m one is not
    xs, ys, _, diag = mesh.mesh_polygon(polygon, 600)
    filled = 1.0 + tiny.area / polygon.area
    assert diag['defeatured_ratio'] == pytest.approx(filled, rel=1e-6)
    assert not shapely.contains_xy(big.buffer(-50), xs, ys).any()

    # refine past it and the small nunatak is honored again
    xs, ys, _, diag = mesh.mesh_polygon(polygon, 120)
    assert diag['defeatured_ratio'] == pytest.approx(1.0)
    assert not shapely.contains_xy(tiny.buffer(-20), xs, ys).any()
    assert not shapely.contains_xy(big.buffer(-50), xs, ys).any()


def test_no_point_ever_lands_on_rock():
    """
    Filling in an unresolvable nunatak makes it ice to the mesh, so an
    element centroid can fall on bedrock. Those have to be dropped: the
    simulation would run them on rock elevation and slope and then
    credit them with the ice area around them.
    """
    outer = geom.Point(0, 0).buffer(3000)
    tiny = geom.Point(500, 500).buffer(60)
    polygon = outer.difference(tiny)

    for h in (600, 300, 120):
        xs, ys, weights, _ = mesh.mesh_polygon(polygon, h)
        assert shapely.contains_xy(polygon, xs, ys).all(), f'h={h}'
        assert np.isclose(weights.sum(), 1.0)
