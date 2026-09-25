"""
Example demonstrating camera sensors with different rendering backends.

Creating cameras as sensors using add_sensor() with three backends
Rasterizer, Raytracer and BatchRenderer.
Test the attachment, add light, batch rendering functionalities.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions, BatchRendererCameraOptions

# The batch renderer compared here requires CUDA, so this example has no CPU backend to offer.
gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

# Try to import LuisaRenderPy to determine if raytracer is available
try:
    import LuisaRenderPy

    ENABLE_RAYTRACER = True
    print("[ok] LuisaRenderPy available - Raytracer will be enabled")
except ImportError:
    ENABLE_RAYTRACER = False
    print("[--] LuisaRenderPy not available - Raytracer will be disabled")

try:
    import gs_madrona

    ENABLE_MADRONA = True
    print("[ok] gs_madrona available - BatchRenderer will be enabled")
except ImportError:
    ENABLE_MADRONA = False
    print("[--] gs_madrona not available - BatchRenderer will be disabled")
ENABLE_MADRONA = ENABLE_MADRONA and (gs.backend == gs.cuda)
# Choose renderer based on raytracer availability
if ENABLE_RAYTRACER:
    renderer = gs.renderers.RayTracer(
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ColorTexture(
                color=(0.2, 0.3, 0.5),
            ),
        ),
        env_radius=20.0,
    )
else:
    # Use Rasterizer as fallback renderer
    renderer = gs.renderers.Rasterizer()

scene = gs.Scene(
    renderer=renderer,
    show_viewer=False,
)

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    surface=gs.surfaces.Rough(
        color=(0.4, 0.4, 0.4),
    ),
)

sphere = scene.add_entity(
    morph=gs.morphs.Sphere(
        radius=0.5,
        pos=(0.0, 0.0, 2.0),
    ),
    surface=gs.surfaces.Smooth(
        color=(1.0, 0.5, 0.5),
    ),
)

box = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.3, 0.3, 0.3),
        pos=(1.0, 1.0, 1.0),
    ),
    surface=gs.surfaces.Rough(
        color=(0.5, 1.0, 0.5),
    ),
)

# Define common camera parameters
CAMERA_COMMON_KWARGS = dict(
    {
        "up": (0.0, 0.0, 1.0),
        "near": 0.1,
        "far": 100.0,
    }
)

CAMERA_SENSORS_KWARGS = [
    {
        "name": "cam0",
        "pos": (3.0, 0.0, 2.0),
        "lookat": (0.0, 0.0, 1.0),
        "fov": 60.0,
        "attachment": None,  # No attachment
        "lights": [{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 1.0}],
    },
    {
        "name": "cam1",
        "pos": (0.0, 3.0, 2.0),
        "lookat": (0.0, 0.0, 1.0),
        "fov": 60.0,
        "attachment": None,
        "lights": [],
    },
    {
        "name": "cam_attached",
        "pos": (0.0, 0.0, 1.0),
        "lookat": (0.0, 0.0, 0.0),
        "fov": 70.0,
        "attachment": {
            "entity_idx": None,
            "link_idx_local": 0,
        },
        "lights": [],
    },
    {
        "name": "cam_offset_T",
        "pos": (0.0, 0.0, 1.0),
        "lookat": (0.0, 0.0, 0.0),
        "fov": 70.0,
        "attachment": {
            "entity_idx": None,
            "link_idx_local": 0,
            "offset_T": np.eye(4),  # Identity transform for testing
        },
        "lights": [],
    },
]


# Create camera configurations for all backends
backends = [
    ("rasterizer", RasterizerCameraOptions, True),  # Always enabled
    ("raytracer", RaytracerCameraOptions, ENABLE_RAYTRACER),
    ("batch_render", BatchRendererCameraOptions, ENABLE_MADRONA),
]

backend_configs = {}
for backend_name, options_class, enabled in backends:
    if not enabled:
        continue

    configs = []
    for camera_config in CAMERA_SENSORS_KWARGS:
        name = f"{backend_name}_{camera_config['name']}"
        res = (500, 600)

        # Create options with common and backend-specific parameters
        options_kwargs = {
            "res": res,
            "pos": camera_config["pos"],
            "lookat": camera_config["lookat"],
            "up": CAMERA_COMMON_KWARGS["up"],
            "fov": camera_config["fov"],
            "lights": camera_config["lights"],
        }

        # Handle attachment
        attachment = camera_config["attachment"]
        if attachment is not None:
            # For attached cameras, set the entity_idx to the sphere's index
            options_kwargs.update(
                {
                    "entity_idx": sphere.idx,
                    "link_idx_local": attachment["link_idx_local"],
                }
            )
            if "offset_T" in attachment:
                options_kwargs["offset_T"] = attachment["offset_T"]

        # Add backend-specific parameters
        if options_class is RasterizerCameraOptions:
            options_kwargs.update({"near": CAMERA_COMMON_KWARGS["near"], "far": CAMERA_COMMON_KWARGS["far"]})
        elif options_class is RaytracerCameraOptions:
            options_kwargs.update(
                {
                    "model": "pinhole",
                    "spp": 64,
                    "denoise": False,
                }
            )
            if attachment is None:  # Only add env surface for non-attached cameras
                options_kwargs.update(
                    {
                        "env_surface": gs.surfaces.Emission(
                            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
                        ),
                        "env_radius": 20.0,
                    }
                )
        elif options_class is BatchRendererCameraOptions:
            options_kwargs.update({"use_rasterizer": True})
            if camera_config["lights"]:
                adjusted_lights = [{**light, "directional": False} for light in camera_config["lights"]]
                options_kwargs["lights"] = adjusted_lights

        # Adjust lights for raytracer (different intensity/color)
        if options_class is RaytracerCameraOptions and camera_config["lights"]:
            adjusted_lights = [
                {**light, "color": (10.0, 10.0, 10.0), "intensity": 1.0} for light in camera_config["lights"]
            ]
            options_kwargs["lights"] = adjusted_lights

        options = options_class(**options_kwargs)
        configs.append(
            {
                "name": name,
                "options": options,
                "attachment": camera_config["attachment"],
            }
        )

    backend_configs[backend_name] = configs


cameras = {}

for group_name, configs in backend_configs.items():
    print(f"\n=== {group_name} Cameras ===")
    for config in configs:
        camera = scene.add_sensor(config["options"])
        cameras[config["name"]] = camera

    print(f"[ok] Created {len(configs)} {group_name.lower()} cameras")


n_envs = 1
scene.build(n_envs=n_envs)

print("\n=== Identifying Attached Cameras ===")

# Identify cameras that are configured to be attached
attached_cameras = []
for group_name, configs in backend_configs.items():
    for config in configs:
        if config["attachment"] is not None:
            camera = cameras[config["name"]]
            attached_cameras.append(camera)
            print(f"[ok] {config['name']} is attached to sphere")

print(f"[ok] Identified {len(attached_cameras)} attached cameras")

os.makedirs("out/camera_as_sensor", exist_ok=True)

for i in range(100):
    scene.step()
    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        camera_data = {}
        for cam_name, camera in cameras.items():
            data = camera.read()
            camera_data[cam_name] = data
            print(f"  {cam_name.replace('_', ' ').title()} RGB shape: {data.rgb.shape}")

        for cam_name, data in camera_data.items():
            rgb_data = data.rgb[0] if data.rgb.ndim > 3 else data.rgb
            suffix = "_env0" if n_envs > 1 else ""
            filename = f"out/camera_as_sensor/{cam_name}{suffix}_step{i:03d}.png"
            plt.imsave(filename, tensor_to_array(rgb_data))
