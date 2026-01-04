# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive demo with the H1 rough terrain environment, but using a custom USD mesh terrain.

功能：
1) 月面效果：黑色天空(弱DomeLight) + “太阳”方向光(DistantLight) + 月壤暗粗糙材质
2) 在天空放一个“地球球体”：
   - 球体自转（Spin）
   - 球体绕场景原点公转（Orbit）
   - 贴图来自本地 jpg
"""

import argparse
import math
import os
import sys

# -----------------------------------------------------------------------------
# IsaacLab app launch
# -----------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip

from isaaclab.app import AppLauncher

# ---------------------------
# 你的月面 USD 路径 & 地球贴图路径
# ---------------------------
MOON_USD_PATH = (
    "/home/img/IsaacLab/source/isaaclab/isaaclab/assets/moon/"
    "Landscape_Tall_Mountain/Tall_Mountain_10cm.usd"  # Tall_Mountain_10cm_scale0.4.usd
)
EARTH_TEX_PATH = (
    "/home/img/IsaacLab/source/isaaclab/isaaclab/assets/moon/"
    "Solarsystemscope_texture_8k_earth_daymap.jpg"
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive H1 demo on a custom USD mesh terrain (moon look).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
import torch

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

TASK = "Isaac-Velocity-Rough-H1-v0"
RL_LIBRARY = "rsl_rl"


# =============================================================================
# 1) 灯光 / 后处理
# =============================================================================
def setup_black_sky_and_sun(stage):
    """
    设置月面风格的光照：
    - DomeLight 很暗：避免完全黑
    - DistantLight 模拟太阳：产生硬阴影
    """
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(30.0)  # 可调：5~80
    dome.CreateColorAttr(Gf.Vec3f(0.02, 0.02, 0.03))

    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(1200.0)  # 可调：300~2000
    sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))

    # 调太阳方向：让阴影更“月面”
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.ClearXformOpOrder()
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 35.0))


def disable_auto_exposure():
    """
    关闭 RTX 自动曝光。
    否则亮的自发光物体会被“压暗/压白”，导致你很难判断 emissive 是否正常。
    """
    s = carb.settings.get_settings()
    s.set("/rtx/post/tonemap/autoExposure/enabled", False)


# =============================================================================
# 2) 工具：查找 mesh、绑定月壤材质
# =============================================================================
def find_first_mesh_under(stage, root_prefix: str):
    """从 stage 里遍历，找 root_prefix 下的第一个 UsdGeom.Mesh。"""
    root_prefix = str(root_prefix)
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = prim.GetPath().pathString
        if p.startswith(root_prefix) and prim.IsA(UsdGeom.Mesh):
            return prim
    return None


def bind_dark_rough_moon_material(stage, target_prim, mat_path="/World/Looks/MoonRegolith"):
    """
    给月面地形绑定一个暗、粗糙、低高光的 UsdPreviewSurface。
    """
    if not target_prim or not target_prim.IsValid():
        return

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.10, 0.10, 0.10))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))

    # 这条在 Kit 里一般可用；若你的 pxr 版本报 ConnectToSource 签名错，再改成 Output-object 连接法
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(target_prim).Bind(material)


# =============================================================================
# 3) 地球球体：建 UV Mesh + 贴图材质（稳定版）
# =============================================================================
def create_uv_sphere_mesh(stage, mesh_path: str, radius: float, rings: int = 48, segments: int = 96):
    """
    生成一个 UV Sphere（UsdGeom.Mesh），并显式写入 st UV（faceVarying）。
    这样可以确保一定存在 UV，不依赖 UsdGeom.Sphere 是否提供 st。
    """
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    points = []
    face_counts = []
    face_indices = []
    st = []

    idx = lambda r, s: r * (segments + 1) + s

    # 顶点
    for r in range(rings + 1):
        v = r / rings
        theta = v * math.pi
        sin_t, cos_t = math.sin(theta), math.cos(theta)

        for s in range(segments + 1):
            u = s / segments
            phi = u * (2.0 * math.pi)
            x = radius * sin_t * math.cos(phi)
            y = radius * sin_t * math.sin(phi)
            z = radius * cos_t
            points.append(Gf.Vec3f(x, y, z))

    # 面 + UV（四边形）
    for r in range(rings):
        for s in range(segments):
            i0 = idx(r, s)
            i1 = idx(r, s + 1)
            i2 = idx(r + 1, s + 1)
            i3 = idx(r + 1, s)

            face_counts.append(4)
            face_indices.extend([i0, i1, i2, i3])

            u0, u1 = s / segments, (s + 1) / segments
            v0, v1 = r / rings, (r + 1) / rings

            # 翻转 V：经纬贴图常用
            st.extend(
                [
                    Gf.Vec2f(u0, 1.0 - v0),
                    Gf.Vec2f(u1, 1.0 - v0),
                    Gf.Vec2f(u1, 1.0 - v1),
                    Gf.Vec2f(u0, 1.0 - v1),
                ]
            )

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)  # 避免背面不可见导致“外面看黑/里面亮”的错觉

    # normals：写 vertex normal，保证渲染稳定
    normals = []
    for p in points:
        n = Gf.Vec3f(p[0], p[1], p[2])
        n.Normalize()
        normals.append(n)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")

    # UV primvar st
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(st)

    return mesh.GetPrim()


def bind_earth_texture_material_stable(
    stage,
    target_prim,
    tex_path: str,
    mat_path="/World/Looks/EarthTexMat",
    emissive_gain: float = 5.0,
):
    """
    稳定版贴图材质：
    - diffuseColor = 原始贴图（不做增益） => 确保“看得见纹理”
    - emissiveColor = 贴图 * gain         => 控制自发光强度
    这样不会一调亮就冲成纯白，也不容易一调小就黑。
    """
    if not target_prim or not target_prim.IsValid():
        return

    material = UsdShade.Material.Define(stage, mat_path)

    # Preview surface
    pbr = UsdShade.Shader.Define(stage, mat_path + "/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbr.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # 读取 UV 的 primvar
    st_reader = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # base texture（不加增益，给 diffuse）
    tex = UsdShade.Shader.Define(stage, mat_path + "/EarthTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # emissive texture（同一张图，但 scale=gain，给 emissive）
    tex_e = UsdShade.Shader.Define(stage, mat_path + "/EarthTexEmissive")
    tex_e.CreateIdAttr("UsdUVTexture")
    tex_e.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
    tex_e.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex_e.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_e.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex_e.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(emissive_gain, emissive_gain, emissive_gain, 1.0))
    tex_e_rgb = tex_e.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # 连接到 PBR
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_e_rgb)

    # material surface <- pbr surface（用 Output 对象连接，兼容旧 pxr）
    pbr_surface_out = pbr.GetOutput("surface")
    if not pbr_surface_out:
        pbr_surface_out = pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    mat_surface_out = material.GetSurfaceOutput()
    if not mat_surface_out:
        mat_surface_out = material.CreateSurfaceOutput()

    mat_surface_out.ConnectToSource(pbr_surface_out)
    UsdShade.MaterialBindingAPI(target_prim).Bind(material)


def spawn_earth_orbit_spin_sphere(
    stage,
    tex_path: str,
    root_path="/World/EarthOrbit",
    orbit_radius=160.0,
    elevation_deg=25.0,
    sphere_radius=24.0,
    emissive_gain=5.0,
):
    """
    生成一个“可公转+自转”的地球球体层级：

    /World/EarthOrbit            （平移：决定公转位置）
      /Spin                      （旋转：决定自转角）
        /Mesh                    （球体 mesh）

    返回：
      orbit_translate_op：每帧改它实现公转
      spin_rotate_op：每帧改它实现自转
      orbit_radius, elevation_deg：方便 update 时使用
    """
    # Orbit Xform (position)
    orbit = UsdGeom.Xform.Define(stage, root_path)
    orbit_xf = UsdGeom.Xformable(orbit.GetPrim())
    orbit_xf.ClearXformOpOrder()
    orbit_translate = orbit_xf.AddTranslateOp()

    # Spin Xform (rotation)
    spin = UsdGeom.Xform.Define(stage, root_path + "/Spin")
    spin_xf = UsdGeom.Xformable(spin.GetPrim())
    spin_xf.ClearXformOpOrder()
    spin_rotate = spin_xf.AddRotateXYZOp()

    # Sphere mesh under spin
    mesh_prim = create_uv_sphere_mesh(stage, root_path + "/Spin/Mesh", radius=sphere_radius)

    # Bind material
    bind_earth_texture_material_stable(stage, mesh_prim, tex_path=tex_path, emissive_gain=emissive_gain)

    return orbit_translate, spin_rotate, orbit_radius, elevation_deg


def raycast_ground_height_z(x: float, y: float, z_start: float = 1000.0, z_end: float = -1000.0):
    """
    用 PhysX raycast 在世界坐标 (x,y) 往下打射线，返回命中的地面 z；没命中返回 None。
    """
    import omni.physx
    from pxr import Gf

    qi = omni.physx.get_physx_scene_query_interface()
    origin = Gf.Vec3f(x, y, z_start)
    direction = Gf.Vec3f(0.0, 0.0, -1.0)
    max_dist = float(z_start - z_end)

    hit = qi.raycast_closest(origin, direction, max_dist)

    # --- 形式 A：对象（最常见） ---
    # 典型字段：hit.hit (bool), hit.position (Gf.Vec3f)
    if hasattr(hit, "hit"):
        if bool(hit.hit):
            # position 可能是 Gf.Vec3f / tuple
            p = hit.position
            return float(p[2])
        return None

    # --- 形式 B：dict ---
    if isinstance(hit, dict):
        if hit.get("hit", False) and "position" in hit:
            return float(hit["position"][2])
        return None

    # --- 形式 C：tuple/list ---
    if isinstance(hit, (tuple, list)) and len(hit) >= 2:
        if isinstance(hit[0], (bool, int)):
            if not bool(hit[0]):
                return None
            return float(hit[1][2])
        if hasattr(hit[0], "__len__") and len(hit[0]) >= 3:
            return float(hit[0][2])

    return None



def lift_robots_above_terrain(env, clearance: float = 0.5, env_ids=None):
    """
    把每个 env 的机器人抬到“地形高度 + clearance”。
    用 root_state_w 只改 z，避免四元数顺序问题；同时把速度清零避免弹飞。
    """
    robot = env.unwrapped.scene["robot"]

    # root_state_w: [x,y,z, q?, vx,vy,vz, wx,wy,wz] 具体内部顺序由 IsaacLab 保证一致
    root_state = robot.data.root_state_w.clone()

    num_envs = root_state.shape[0]
    device = root_state.device

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    # 只处理选定 env
    for k, eid in enumerate(env_ids.tolist()):
        x = float(root_state[eid, 0].item())
        y = float(root_state[eid, 1].item())
        z_now = float(root_state[eid, 2].item())

        # 从当前高度上方一点开始往下打，避免起点在模型内部
        ground_z = raycast_ground_height_z(x, y, z_start=z_now + 50.0, z_end=-1000.0)

        if ground_z is None:
            continue

        # 防呆：如果命中点比当前还高很多，可能打到机器人/别的物体了，先跳过
        if ground_z > z_now + 5.0:
            continue

        root_state[eid, 2] = ground_z + clearance

        # 速度清零（root_state 后半部分通常是 linvel+angvel；这里用更稳的 API）
        try:
            pass
        except Exception:
            pass

    # ✅ 关键：写回仿真必须在 inference_mode（你之前那个 RuntimeError 就是这里）
    with torch.inference_mode():
        # 更稳：直接写 root_state
        robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)

        # 再保险：把 root velocity 清零（有的版本 root_state 写回不覆盖 vel）
        try:
            zeros = torch.zeros_like(robot.data.root_vel_w)
            robot.write_root_velocity_to_sim(zeros[env_ids], env_ids=env_ids)
        except Exception:
            pass




# =============================================================================
# 4) Demo
# =============================================================================
class H1RoughDemo:
    """Interactive demo for the H1 environment (USD terrain + moon look)."""

    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

        env_cfg = H1RoughEnvCfg_PLAY()

        # ---------------------------
        # 基本环境配置
        # ---------------------------
        env_cfg.scene.num_envs = 25
        env_cfg.scene.env_spacing = 5.0
        env_cfg.curriculum = None
        env_cfg.episode_length_s = 1000000
        env_cfg.sim.gravity = (0.0, 0.0, -1.62)  # 月球重力

        # 使用 USD mesh 地形
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = MOON_USD_PATH
        env_cfg.scene.terrain.use_terrain_origins = False
        env_cfg.scene.terrain.env_spacing = 0.0
        env_cfg.scene.terrain.debug_vis = True

        # 尝试关闭默认 skylight（不同版本可能字段不同）
        try:
            env_cfg.scene.sky_light = None
        except Exception:
            pass

        # 速度命令范围（保留你的设置）
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)

        # 初始高度抬高，避免卡进地形
        try:
            x, y, _ = env_cfg.scene.robot.init_state.pos
            env_cfg.scene.robot.init_state.pos = (x, y, 5.0)
        except Exception:
            pass

        # wrap env
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        # load trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        # camera + keyboard
        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()

        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

        # Earth motion state
        self._earth_spin_deg = 0.0
        self._earth_orbit_deg = 0.0
        self._earth_spin_speed = 7.5   # 度/秒，自转速度
        self._earth_orbit_speed = 2.0  # 度/秒，公转速度

        self._post_setup_stage()

    def _post_setup_stage(self):
        stage = get_current_stage()

        # 灯光 + 关闭曝光
        setup_black_sky_and_sun(stage)
        disable_auto_exposure()

        # 生成地球：公转 + 自转
        self._earth_orbit_translate_op, self._earth_spin_rotate_op, self._earth_orbit_r, self._earth_elev_deg = (
            spawn_earth_orbit_spin_sphere(
                stage,
                tex_path=EARTH_TEX_PATH,
                root_path="/World/EarthOrbit",
                orbit_radius=160.0,     # 你场地 40m，建议 120~240
                elevation_deg=25.0,     # 抬角 10~35
                sphere_radius=24.0,     # 球直径约 48m，看起来会很大；想小就 8~16
                emissive_gain=5.0,      # 先 1~10 调；大了会冲白
            )
        )

        # 月面地形材质
        terrain_root = "/World/ground/terrain"
        mesh_prim = stage.GetPrimAtPath(terrain_root + "/Tall_Mountain_10cm/geometry/mesh")
        if not mesh_prim or not mesh_prim.IsValid():
            mesh_prim = find_first_mesh_under(stage, terrain_root)

        if mesh_prim and mesh_prim.IsValid():
            print("[INFO] Moon terrain mesh prim:", mesh_prim.GetPath())
            bind_dark_rough_moon_material(stage, mesh_prim)
        else:
            print("[WARN] Could not find a Mesh under:", terrain_root, " -> skip moon material.")

    def update_earth_motion(self, dt: float):
        """
        每帧更新地球的自转 + 公转：
        - 自转：改 /World/EarthOrbit/Spin 的 RotateXYZ
        - 公转：改 /World/EarthOrbit 的 Translate
        """
        self._earth_spin_deg = (self._earth_spin_deg + self._earth_spin_speed * dt) % 360.0
        self._earth_orbit_deg = (self._earth_orbit_deg + self._earth_orbit_speed * dt) % 360.0

        az = math.radians(self._earth_orbit_deg)
        el = math.radians(self._earth_elev_deg)

        x = self._earth_orbit_r * math.cos(el) * math.cos(az)
        y = self._earth_orbit_r * math.cos(el) * math.sin(az)
        z = self._earth_orbit_r * math.sin(el)

        self._earth_orbit_translate_op.Set(Gf.Vec3d(x, y, z))
        self._earth_spin_rotate_op.Set(Gf.Vec3f(0.0, self._earth_spin_deg, 0.0))

    # ---------------- Camera & input ----------------
    def create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"

        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.1, 5000.0))

        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))

        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        T = 1.0
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()

        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
            return

        if len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
            return

        prim_splitted_path = selected_prim_paths[0].split("/")
        if len(prim_splitted_path) >= 4 and prim_splitted_path[3].startswith("env_"):
            self._selected_id = int(prim_splitted_path[3][4:])
            if self._previous_selected_id != self._selected_id:
                self.viewport.set_active_camera(self.camera_path)
            self._update_camera()
        else:
            print("The selected prim was not a H1 robot")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    demo_h1 = H1RoughDemo()
    obs, _ = demo_h1.env.reset()


    for _ in range(2):
        with torch.inference_mode():
            a = demo_h1.policy(obs)
            obs, _, _, _ = demo_h1.env.step(a)

    lift_robots_above_terrain(demo_h1.env, clearance=1.3)

    # 更稳：优先用环境的 step_dt
    if hasattr(demo_h1.env.unwrapped, "step_dt"):
        dt = float(demo_h1.env.unwrapped.step_dt)
    else:
        dt = 0.02

    while simulation_app.is_running():
        demo_h1.update_selected_object()
        demo_h1.update_earth_motion(dt)

        # ✅ 只包 policy
        with torch.inference_mode():
            action = demo_h1.policy(obs)

        # ✅ env.step 不要在 inference_mode 里
        obs, rew, dones, infos = demo_h1.env.step(action)
        obs[:, 9:13] = demo_h1.commands

        # ✅ 如果发生 reset（dones=True），立刻把这些 env 的机器人抬到地形上方
        if torch.any(dones):
            reset_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1).to(dtype=torch.long)
            lift_robots_above_terrain(demo_h1.env, clearance=1.3, env_ids=reset_ids)



if __name__ == "__main__":
    main()
    simulation_app.close()


# how to get usd
'''
python scripts/tools/convert_mesh.py \
    --input source/isaaclab/isaaclab/assets/moon/Landscape_Tall_Mountain/Tall_Mountain_10cm.stl \
    --output source/isaaclab/isaaclab/assets/moon/Landscape_Tall_Mountain/Tall_Mountain_10cm_scale0.4.usd \
    --collision-approximation triangleMesh \
    --scale 0.4
'''