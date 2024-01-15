import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xdatasets as xd

import xhydro as xh


class TestWatershedDelineation:
    m = leafmap.Map(center=(48.63, -74.71), zoom=5, basemap="USGS Hydrography")

    @pytest.mark.parametrize(
        "lng_lat, area",
        [
            ((-73.118597, 46.042467), 23933972552.885937),
            ((-66.153789, 50.265321), 18891676494.940426),
        ],
    )
    def test_watershed_delineation_from_coords(self, lng_lat, area):
        gdf = xh.gis.watershed_delineation(coordinates=lng_lat)
        np.testing.assert_almost_equal([gdf.to_crs(32198).area.values[0]], [area])

    @pytest.mark.parametrize("area", [(18891676494.940426)])
    def test_watershed_delineation_from_map(self, area):
        # Richelieu watershed
        self.m.draw_features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-66.153789, 50.265321]},
            }
        ]
        gdf = xh.gis.watershed_delineation(map=self.m)
        np.testing.assert_almost_equal([gdf.to_crs(32198).area.values[0]], [area])


class TestWatershedOperations:
    gdf = xd.Query(
        **{
            "datasets": {
                "deh_polygons": {
                    "id": ["031501", "042103"],
                    "regulated": ["Natural"],
                }
            }
        }
    ).data.reset_index()

    def test_watershed_properties(self):
        _properties_name = ["area", "perimeter", "gravelius", "centroid"]
        properties_values = np.array(
            [
                [
                    21868619.035204668,
                    27186.996844559395,
                    1.6400067113344199,
                    (-72.48631199105834, 46.22277542928622),
                ],
                [
                    579479639.8084792,
                    283765.05839030433,
                    3.3253311032394937,
                    (-78.37036445281987, 46.48287117609677),
                ],
            ],
            dtype="object",
        )

        gdf_properties = xh.gis.watershed_properties(self.gdf)

        np.testing.assert_array_equal(
            gdf_properties[_properties_name].values, properties_values
        )

    @pytest.fixture
    def land_classification_data(self):
        data = {
            "pct_crops": {"031501": 0.7761508991718495, "042103": 0.0},
            "pct_built_area": {
                "031501": 0.030159065706857738,
                "042103": 0.00010067694852579148,
            },
            "pct_trees": {"031501": 0.1916484013692483, "042103": 0.8636022653195444},
            "pct_rangeland": {
                "031501": 0.002041633752044415,
                "042103": 0.026126172157203968,
            },
            "pct_water": {"031501": 0.0, "042103": 0.10998710919246692},
            "pct_bare_ground": {"031501": 0.0, "042103": 2.142062734591308e-05},
            "pct_flooded_vegetation": {"031501": 0.0, "042103": 0.00016197774384218392},
            "pct_snow/ice": {"031501": 0.0, "042103": 3.780110708102308e-07},
        }
        df = pd.DataFrame.from_dict(data)
        df.index.name = "Station"
        return df

    def test_land_classification(self, land_classification_data):
        df = xh.gis.land_use_classification(self.gdf, unique_id="Station")

        pd.testing.assert_frame_equal(df, land_classification_data)

    def _plot_fn(self):
        xh.gis.land_use_plot(self.gdf, unique_id="Station", idx=1)

        yield plt.show()
        plt.close("all")

    def test_land_classification_plot(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        self._plot_fn()
