from simcore import RobotSystem, Pose, load_yaml

cfg = load_yaml("configs/global_config.yaml")
system = RobotSystem(cfg)

system.run()