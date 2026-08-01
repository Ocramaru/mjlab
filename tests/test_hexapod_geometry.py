"""Geometry regression tests for the Freenove hexapod MJCF.

Every expected value here was measured from the manufacturer's STEP assembly
(``thirdparty/hexapod-cad/big_hexapod_freenove.STEP``) by locating the servo
horns, which sit on the servo output shafts and therefore mark the joint axes.
Offsets were solved across all six legs so that fixed linkage structure could
be separated from the arbitrary pose the assembly was saved in.

These tests exist so that adding the remaining legs cannot silently perturb
the geometry we verified for leg 2.
"""

import mujoco
import numpy as np
import pytest

from mjlab import MJLAB_SRC_PATH

HEXAPOD_XML = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "hexapod" / "xmls" / "hexapod.xml"
)

# Joint-axis positions relative to the base body, at the zero pose, in mm.
#
# Two frame changes are applied to the raw CAD measurements:
#   1. Yaw:  forward is -90 deg in CAD (the ultrasonic sensor and camera tower
#            sit there), so robot yaw = CAD yaw + 90, i.e. (x, y) -> (-y, x).
#   2. Height: the body origin is the chassis box centre, which is the CAD
#            origin shifted down 33.8 mm, so z_robot = z_cad + 33.8.
CAD_JOINT_AXES_MM = {
  "leg2_coxa": (0.0, 85.0, 31.8),  # CAD (85.0, 0.0, -2.0)
  "leg2_femur": (-11.8, 118.3, -7.5),  # CAD (118.3, 11.8, -41.3)
  "leg2_tibia": (-11.8, 208.3, -7.5),  # CAD (208.3, 11.8, -41.3)
}

# Height of the coxa axes above the body origin, mm.
HIP_HEIGHT_MM = 31.8

# Link lengths in mm, constant across all six legs in the CAD.
FEMUR_LENGTH_MM = 90.0
TIBIA_LENGTH_MM = 110.0

TOLERANCE_MM = 0.5


@pytest.fixture
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(HEXAPOD_XML))


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
  d = mujoco.MjData(model)
  d.qpos[:] = 0.0
  d.qpos[3] = 1.0  # identity quaternion for the freejoint
  mujoco.mj_forward(model, d)
  return d


def body_pos_mm(model, data, name: str) -> np.ndarray:
  """Body origin relative to the base body, in millimetres."""
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  return (data.xpos[bid] - data.xpos[base]) * 1000.0


@pytest.mark.parametrize("name,expected", CAD_JOINT_AXES_MM.items())
def test_joint_axis_matches_cad(model, data, name, expected):
  measured = body_pos_mm(model, data, name)
  error = np.linalg.norm(measured - np.array(expected))
  assert error < TOLERANCE_MM, f"{name}: {measured.round(2)} vs CAD {expected}"


def test_link_lengths(model, data):
  femur = body_pos_mm(model, data, "leg2_femur")
  tibia = body_pos_mm(model, data, "leg2_tibia")
  assert np.linalg.norm(tibia - femur) == pytest.approx(
    FEMUR_LENGTH_MM, abs=TOLERANCE_MM
  )

  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg2_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  foot = (data.site_xpos[sid] - data.xpos[base]) * 1000.0
  assert np.linalg.norm(foot - tibia) == pytest.approx(
    TIBIA_LENGTH_MM, abs=TOLERANCE_MM
  )


def set_joint(model, data, name: str, degrees: float) -> None:
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  data.qpos[model.jnt_qposadr[jid]] = np.radians(degrees)
  mujoco.mj_forward(model, data)


def foot_mm(model, data) -> np.ndarray:
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg2_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  return (data.site_xpos[sid] - data.xpos[base]) * 1000.0


def test_femur_pitch_matches_cad_pose(model, data):
  """The CAD assembly has leg 2's femur pitched up 5.04 deg.

  Driving our model to that angle should reproduce the measured tibia axis,
  which checks the chain end-to-end rather than link by link.
  """
  set_joint(model, data, "leg2_femur_joint", 5.04)
  measured = body_pos_mm(model, data, "leg2_tibia")
  cad_posed = np.array([-11.8, 207.6, 0.4])  # CAD (207.6, 11.8, -33.4)
  assert np.linalg.norm(measured - cad_posed) < TOLERANCE_MM


LEG_LAYOUT = {  # leg: (hip radius mm, robot yaw deg, side sign)
  1: (93.8, 144.0, +1),
  2: (85.0, 90.0, +1),
  3: (93.8, 36.0, +1),
  4: (93.8, -36.0, -1),
  5: (85.0, -90.0, -1),
  6: (93.8, -144.0, -1),
}


@pytest.mark.parametrize("leg,expected", LEG_LAYOUT.items())
def test_hip_placement(model, data, leg, expected):
  radius, yaw, _ = expected
  hip = body_pos_mm(model, data, f"leg{leg}_coxa")
  assert np.hypot(hip[0], hip[1]) == pytest.approx(radius, abs=TOLERANCE_MM)
  assert np.degrees(np.arctan2(hip[1], hip[0])) == pytest.approx(yaw, abs=0.1)
  assert hip[2] == pytest.approx(HIP_HEIGHT_MM, abs=TOLERANCE_MM)


@pytest.mark.parametrize("leg", LEG_LAYOUT)
def test_every_leg_has_identical_links(model, data, leg):
  """All six legs are the same physical assembly; only placement differs."""
  femur = body_pos_mm(model, data, f"leg{leg}_femur")
  tibia = body_pos_mm(model, data, f"leg{leg}_tibia")
  coxa = body_pos_mm(model, data, f"leg{leg}_coxa")
  assert np.linalg.norm(femur - coxa) == pytest.approx(52.85, abs=TOLERANCE_MM)
  assert np.linalg.norm(tibia - femur) == pytest.approx(90.0, abs=TOLERANCE_MM)


def test_left_and_right_legs_are_mirrored(model, data):
  """Mirror pairs 1<->6, 2<->5, 3<->4 must reflect through the xz plane."""
  for left, right in ((1, 6), (2, 5), (3, 4)):
    lf = body_pos_mm(model, data, f"leg{left}_femur")
    rf = body_pos_mm(model, data, f"leg{right}_femur")
    assert lf[0] == pytest.approx(rf[0], abs=TOLERANCE_MM)
    assert lf[1] == pytest.approx(-rf[1], abs=TOLERANCE_MM)
    assert lf[2] == pytest.approx(rf[2], abs=TOLERANCE_MM)


@pytest.mark.parametrize("leg", LEG_LAYOUT)
def test_positive_femur_lifts_every_leg(model, data, leg):
  """Option A: one sign convention across all six legs, including mirrored ones."""
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"leg{leg}_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  before = (data.site_xpos[sid] - data.xpos[base])[2]
  set_joint(model, data, f"leg{leg}_femur_joint", 30.0)
  assert (data.site_xpos[sid] - data.xpos[base])[2] > before


def test_total_mass_matches_cad(model):
  """Mass and inertia are solved from the CAD, calibrated against the scale.

  Measured on the real robot: MG995 servo 54.2 g with its cable, matching
  the published 55 g to within scale precision; and the acrylic plates at
  1.005 g/cm3 effective (nominal 1.18, less sheet-thickness tolerance).  Steel is 6.73 g/cm3 effective (M4 nut 0.74 g over its CAD volume), brass
  standoffs nominal 8.5.  Screws are point masses at their CAD positions,
  and every servo mount is topped up to the measured 8.72 g of hardware.
  Four 48 g cells sit at the battery-holder positions, and an 80 g wiring
  harness reconciles the chassis against the assembled robot: 1900 g without
  batteries on the scale, so 2091 g with them.  The legs are pinned
  independently -- the tibia body is 71.6 g against 71.43 g measured.
  """
  assert model.body_mass.sum() == pytest.approx(2.091, abs=2e-3)
