import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import threading
import time
import numpy as np
import mujoco as mj
import mujoco.viewer
from src.common.utils import load_yaml
from typing import Dict, List, Optional, Tuple


class DeviceInfo:
    """Stores metadata about a device (robot/sensor/object)."""
    
    def __init__(self, name: str, device_type: str, prefix: str):
        self.name = name
        self.type = device_type
        self.prefix = prefix  # Prefix used in MuJoCo (e.g., "arm_right/")
        
        # Joint information
        self.joint_ids: List[int] = []
        self.joint_names: List[str] = []
        self.dof_ids: List[int] = []  # Degree of freedom IDs (for qpos/qvel)
        
        # Actuator information
        self.actuator_ids: List[int] = []
        self.actuator_names: List[str] = []
        
        # Body information
        self.body_ids: List[int] = []
        self.body_names: List[str] = []
        
        # Initial configuration
        self.q0: Optional[np.ndarray] = None
        

class SimulationModel:
    """
    Threaded MuJoCo physics simulation with dynamic model composition.
    Provides thread-safe interface via buffers.
    """

    def __init__(self, config: dict):
        self.config = config
        
        # Build composite model from config
        self.mj_model, self.devices, self.objects = self._build_model_from_config(config)
        self.mj_data = mj.MjData(self.mj_model)
        
        # Initialize device positions
        self._set_initial_positions()
        
        # Viewer settings
        self.viewer = None
        self.use_viewer = config.get('use_viewer', False)
        
        # Threading
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.running = False
        self.dt = self.mj_model.opt.timestep

    def _build_model_from_config(self, config: dict) -> Tuple[mj.MjModel, Dict[str, DeviceInfo], Dict[str, DeviceInfo]]:
        """
        Build MuJoCo model by composing base world + devices + objects.
        """
        # Load base world as MjSpec (editable)
        world_spec = mj.MjSpec.from_file(config['world_model'])
        world_spec.copy_during_attach = True  # Allow using same model multiple times
        
        devices, objects = {}, {}
        
        # Add devices (robots, sensors)
        for device_cfg in config.get('devices', []):
            if not device_cfg.get('enabled', True):
                continue
                
            name = device_cfg['name']
            device_type = device_cfg['type']
            model_path = device_cfg['model_path']
            base_pose = device_cfg.get('base_pose', {})
            
            # Load device model
            device_spec = mj.MjSpec.from_file(model_path)
            
            # Extract pose
            pos = base_pose.get('position', [0, 0, 0])
            quat = base_pose.get('orientation', [1, 0, 0, 0])  # [w, x, y, z]
            
            # Create attachment frame in world
            attach_frame = world_spec.worldbody.add_frame(pos=pos, quat=quat)
            
            # FIX: Attach device with "/" in prefix
            world_spec.attach(device_spec, frame=attach_frame, prefix=f"{name}/")  # <-- CHANGED
            
            # Store device info
            device_info = DeviceInfo(name, device_type, prefix=f"{name}/")
            device_info.q0 = np.array(device_cfg.get('q0', []))
            devices[name] = device_info
        
        # Add objects (static/dynamic props)
        for obj_cfg in config.get('objects', []):
            if not obj_cfg.get('enabled', True):
                continue
                
            name = obj_cfg['name']
            obj_type = obj_cfg['type']
            model_path = obj_cfg['model_path']
            pose = obj_cfg.get('pose', {})
            
            # Load object model
            obj_spec = mj.MjSpec.from_file(model_path)
            
            # Extract pose
            pos = pose.get('position', [0, 0, 0])
            quat = pose.get('orientation', [1, 0, 0, 0])  # [w, x, y, z]
            
            # Create attachment frame in world
            attach_frame = world_spec.worldbody.add_frame(pos=pos, quat=quat)
            
            # FIX: Attach object with "/" in prefix
            world_spec.attach(obj_spec, frame=attach_frame, prefix=f"{name}/")  # <-- CHANGED
            
            # Store object info
            obj_info = DeviceInfo(name, obj_type, prefix=f"{name}/")
            objects[name] = obj_info 
        
        # Compile final model
        compiled_model = world_spec.compile()
        
        # Extract IDs from compiled model
        all_entities = {**devices, **objects}
        self._extract_device_ids(compiled_model, all_entities)
        
        return compiled_model, devices, objects  # <-- FIXED return type
    
    def _extract_device_ids(self, model: mj.MjModel, devices: Dict[str, DeviceInfo]):
        """
        Extract joint, actuator, and body IDs for each device.
        """
        for device_name, device_info in devices.items():
            prefix = device_info.prefix
            
            # Find joints
            for joint_id in range(model.njnt):
                joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name and joint_name.startswith(prefix):
                    device_info.joint_ids.append(joint_id)
                    device_info.joint_names.append(joint_name)
                    
                    # Get corresponding DOF ID (qpos/qvel index)
                    dof_adr = model.jnt_dofadr[joint_id]
                    device_info.dof_ids.append(dof_adr)
            
            # Find actuators
            for act_id in range(model.nu):
                act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id)
                if act_name and act_name.startswith(prefix):
                    device_info.actuator_ids.append(act_id)
                    device_info.actuator_names.append(act_name)
            
            # Find bodies
            for body_id in range(model.nbody):
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
                if body_name and body_name.startswith(prefix):
                    device_info.body_ids.append(body_id)
                    device_info.body_names.append(body_name)
    
    def _set_initial_positions(self):
        """Set initial joint positions and control targets from config."""
        for device_info in self.devices.values():
            if device_info.q0 is not None and len(device_info.q0) > 0:
                # Set qpos for all DOFs we have q0 for
                n_qpos = min(len(device_info.q0), len(device_info.dof_ids))
                for i in range(n_qpos):
                    dof_id = device_info.dof_ids[i]
                    self.mj_data.qpos[dof_id] = device_info.q0[i]
                
                # Set ctrl for all actuators we have q0 for
                n_ctrl = min(len(device_info.q0), len(device_info.actuator_ids))
                for i in range(n_ctrl):
                    act_id = device_info.actuator_ids[i]
                    act_gaintype = self.mj_model.actuator_gaintype[act_id]
                    
                    # For position/servo actuators, set target to match q0
                    if act_gaintype in [mj.mjtGain.mjGAIN_FIXED, mj.mjtGain.mjGAIN_AFFINE]:
                        self.mj_data.ctrl[act_id] = device_info.q0[i]
                    else:
                        # For torque actuators, zero torque
                        self.mj_data.ctrl[act_id] = 0.0
                
                # Set any remaining actuators to zero
                for i in range(n_ctrl, len(device_info.actuator_ids)):
                    self.mj_data.ctrl[device_info.actuator_ids[i]] = 0.0
        
        # Forward kinematics to update all derived quantities
        mj.mj_forward(self.mj_model, self.mj_data)

    # ========================
    # PUBLIC INTERFACE
    # ========================
    
    def start(self, with_viewer: bool = False):
        """Start physics thread, optionally with visualization."""
        self.use_viewer = with_viewer or self.config.get('use_viewer', False)
        self.running = True
        
        if self.use_viewer:
            # Launch viewer in main thread (MuJoCo requirement)
            self._run_with_viewer()
        else:
            # Start physics in background thread
            self.physics_thread.start()

    def stop(self):
        """Stop physics thread."""
        self.running = False
        if self.physics_thread.is_alive():
            self.physics_thread.join()
        if self.viewer is not None:
            self.viewer.close()
    
    def get_device_names(self) -> List[str]:
        """Get list of all device names."""
        return list(self.devices.keys())
    
    def get_device_info(self, device_name: str) -> DeviceInfo:
        """Get device metadata."""
        if device_name not in self.devices:
            raise ValueError(f"Device '{device_name}' not found")
        return self.devices[device_name]
    
    def get_joint_positions(self, device_name: str) -> np.ndarray:
        """Get joint positions for a device."""
        device = self.get_device_info(device_name)
        return self.mj_data.qpos[device.dof_ids].copy()
    
    def get_joint_velocities(self, device_name: str) -> np.ndarray:
        """Get joint velocities for a device."""
        device = self.get_device_info(device_name)
        return self.mj_data.qvel[device.dof_ids].copy()
    
    def set_control_input(self, device_name: str, values: np.ndarray):
        """
        Set control input (torque/position) for a device.
        """
        device = self.get_device_info(device_name)
        
        if len(values) != len(device.actuator_ids):
            raise ValueError(
                f"Expected {len(device.actuator_ids)} control values, got {len(values)}"
            )
        
        for i, act_id in enumerate(device.actuator_ids):
            self.mj_data.ctrl[act_id] = values[i]
    
    def get_body_pose(self, device_name: str, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pose of a specific body in a device.
        """
        device = self.get_device_info(device_name)
        full_body_name = f"{device.prefix}{body_name}"
        
        body_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id == -1:
            raise ValueError(f"Body '{full_body_name}' not found")
        
        pos = self.mj_data.xpos[body_id].copy()
        quat = self.mj_data.xquat[body_id].copy()  # [w, x, y, z]
        
        return pos, quat
    
    def print_device_info(self):
        """Debug: Print information about all devices."""
        print("\n" + "="*60)
        print("LOADED DEVICES")
        print("="*60)
        
        for name, device in self.devices.items():
            print(f"\n{name} ({device.type}):")
            print(f"  Prefix: {device.prefix}")
            print(f"  Joints: {len(device.joint_ids)}")
            for jname in device.joint_names:
                print(f"    - {jname}")
            print(f"  Actuators: {len(device.actuator_ids)}")
            for aname in device.actuator_names:
                print(f"    - {aname}")
            print(f"  Bodies: {len(device.body_ids)}")
            if device.q0 is not None:
                print(f"  Initial config (q0): {device.q0}")
        
        print("\n" + "="*60 + "\n")

    # ========================
    # PRIVATE
    # ========================
    
    def _run_with_viewer(self):
        """Run physics loop with interactive viewer (blocks main thread)."""
        self.viewer = mj.viewer.launch(self.mj_model, self.mj_data)
        
        while self.running and self.viewer.is_running():
            # Step physics
            mj.mj_step(self.mj_model, self.mj_data)
            
            # Sync viewer
            self.viewer.sync()
            
            # Sleep to match timestep
            time.sleep(self.dt)

    def _physics_loop(self):
        """Headless physics loop (runs in thread)."""
        last_time = time.time()
        
        while self.running:
            # Step physics
            mj.mj_step(self.mj_model, self.mj_data)
            
            # Maintain fixed timestep
            elapsed = time.time() - last_time
            sleep_time = self.dt - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            last_time = time.time()


if __name__ == "__main__":
    cfg = load_yaml("configs/scene_config.yaml")
    sim = SimulationModel(config=cfg)
    
    # Print device information
    sim.print_device_info()
    
    for device_name in sim.get_device_names():
        q = sim.get_joint_positions(device_name)
        device = sim.get_device_info(device_name)
        ctrl = sim.mj_data.ctrl[device.actuator_ids]
        print(f"  {device_name}:")
        print(f"    qpos: {q}")
        print(f"    ctrl: {ctrl}")
        print(f"    q0:   {device.q0}")
    
    # Test: Set some control input
    print("\nSetting control inputs...")
    for device_name in sim.get_device_names():
        device = sim.get_device_info(device_name)
        if len(device.actuator_ids) > 0:
            # Zero torque
            sim.set_control_input(device_name, np.zeros(len(device.actuator_ids)))
    
    # Start with viewer
    try:
        print("\nStarting simulation with viewer...")
        sim.start(with_viewer=True)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        sim.stop()