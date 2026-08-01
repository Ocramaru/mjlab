from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

CARTPOLE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "cartpole" / "xmls" / "cartpole.xml"
)
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


def get_cartpole_robot_cfg() -> EntityCfg:
  """Get a fresh CartPole robot configuration instance."""
  actuators = (
    XmlActuatorCfg(
      target_names_expr=("slide",),
    ),
  )
  articulation = EntityArticulationInfoCfg(actuators=actuators)
  return EntityCfg(spec_fn=get_spec, articulation=articulation)


# if __name__ == "__main__":
#   import mujoco.viewer as viewer
#   robot = Entity(get_cartpole_robot_cfg())
#   viewer.launch(robot.spec.compile())
