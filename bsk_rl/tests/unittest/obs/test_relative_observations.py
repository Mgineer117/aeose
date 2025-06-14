from unittest.mock import MagicMock

import numpy as np
import pytest

from bsk_rl.obs import RelativeProperties


class TestRelativeProperties:
    def test_init(self):
        ob = RelativeProperties(
            dict(name="prop1", fn=lambda d, c: 0.0),
            dict(name="prop2", fn=lambda d, c: 0.0, norm=2.0),
            chief_name="chief",
        )
        assert ob.rel_properties[0]["norm"] == 1.0
        assert len(ob.rel_properties) == 2

    @pytest.mark.parametrize("order", [0, 1])
    def test_identify_chief(self, order):
        ob = RelativeProperties(dict(prop="r_DC_N"), chief_name="chief")
        deputy = MagicMock()
        deputy.name = "deputy"
        chief = MagicMock()
        chief.name = "chief"

        if order == 0:
            deputy.simulator.satellites = [deputy, chief]
        elif order == 1:
            deputy.simulator.satellites = [chief, deputy]
        ob.satellite = deputy
        ob.reset_post_sim_init()
        assert ob.chief == chief

    def test_defined_prop(self):
        ob = RelativeProperties(dict(prop="r_DC_N"), chief_name="chief")
        deputy = MagicMock()
        deputy.dynamics.r_BN_N = np.array([1.0, 2.0, 3.0])
        chief = MagicMock()
        chief.dynamics.r_BN_N = np.array([1.0, 2.0, 3.0])
        ob.satellite = deputy
        ob.chief = chief
        assert np.allclose(ob.get_obs()["r_DC_N"], np.array([0.0, 0.0, 0.0]))

    def test_custom_prop(self):
        ob = RelativeProperties(
            dict(name="prop", fn=lambda d, c: d.prop + c.prop), chief_name="chief"
        )
        deputy = MagicMock(prop=1.0)
        chief = MagicMock(prop=2.0)
        ob.satellite = deputy
        ob.chief = chief
        assert ob.get_obs() == {"prop": 3.0}
