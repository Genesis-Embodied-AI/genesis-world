import genesis as gs


def main():
    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(8.5, 0.0, 4.5),
            camera_lookat=(3.0, 0.0, 0.5),
            camera_fov=50,
        ),
        renderer=gs.renderers.RayTracer(  # type: ignore
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        ),
    )

    # floor
    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -0.5),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )

    # user specified external color texture
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -3, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    # user specified color (using color shortcut)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -1.8, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # smooth shortcut
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -0.6, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    # Iron
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 0.6, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Gold
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 1.8, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Glass
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 3.0, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Opacity
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(2.0, -3, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Smooth(
            color=(1.0, 1.0, 1.0, 0.5),
        ),
    )
    # asset's own attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -2.3, 0.0),
        ),
        material=gs.materials.Kinematic(),
    )
    # override asset's attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -1.0, 0.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    cam_0 = scene.add_camera(
        res=(1600, 900),
        pos=(8.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.7),
        fov=60,
        GUI=True,
        spp=512,
    )
    scene.build()
    scene.reset()
    horizon = 10

    for i in range(horizon):
        scene.step()
        cam_0.render()


if __name__ == "__main__":
    main()
