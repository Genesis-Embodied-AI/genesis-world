import argparse
import asyncio
import logging
import os
import time


def omni_bootstrap(device=0, log_level="warning"):
    import omni.kit_app

    app = omni.kit_app.KitApp()
    kit_dir = os.path.dirname(os.path.abspath(os.path.realpath(omni.kit_app.__file__)))
    kit_path = os.path.join(kit_dir, "apps", "omni.app.empty.kit")

    # The distiller reads the material from the RTX render engine attached to the stage, which a bare hydra texture
    # feeds without the viewport window.
    omni_extensions = [
        "omni.kit.usd.collect",
        "omni.mdl.distill_and_bake",
        "omni.hydra.rtx",
        "omni.kit.hydra_texture",
    ]
    app_args = [
        kit_path,
        "--/app/window/hideUi=True",
        f"--/app/tokens/exe-path={kit_dir}",
        "--/app/enableStdoutOutput=False",  # Disable print outs (print_and_log) on extension startup information
        "--/app/runLoops/present/rateLimitFrequency=60",
        "--/app/vulkan=True",
        "--/app/asyncRendering=False",
        "--/app/python/interceptSysStdOutput=False",
        "--/app/python/logSysStdOutput=False",
        "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
        f"--/omni/log/level={log_level}",
        "--/log/file=",  # Empty string means no log file
        f"--/log/level={log_level}",
        f"--/renderer/activeGpu={device}",
        "--/renderer/enabled=rtx",
        "--/renderer/active=rtx",
        "--/renderer/multiGpu/enabled=False",  # Avoids unnecessary GPU context initialization
        # The bake uses no NGX feature (DLSS), whose initialization fails on some headless datacenter GPUs
        "--/ngx/enabled=False",
        "--no-window",
        "--portable",  # TODO: set portable root to avoid extension conflicts
    ]
    for extension in omni_extensions:
        app_args += ["--enable", extension]
    app.startup(app_args)
    app.update()  # important
    return app


def bake_usd_material(input_file, output_dir, usd_material_paths, device=0, log_level="error"):
    logs = []

    # bootstrap
    start_time = time.time()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    app = omni_bootstrap(device, log_level)
    logs.append(f"\tBootstrap: {time.time() - start_time}, App status: {app.is_running()}.")

    import carb
    import omni.usd
    import omni.mdl.distill_and_bake
    import omni.kit.hydra_texture
    import omni.kit.usd.collect
    from pxr import Usd, UsdGeom

    # open stage
    start_time = time.time()
    omni.usd.get_context().open_stage(input_file)
    logs.append(f"\tOpen stage: {time.time() - start_time}s.")

    # create render product
    start_time = time.time()
    stage = omni.usd.get_context().get_stage()
    # The render engine syncs the materials of the stage only once a render product reads from it, which is bound to a
    # local to stay alive until the export.
    omni.usd.create_hydra_engine("rtx", omni.usd.get_context())
    # The render product views the stage through a camera, which it would otherwise define on the root layer that the
    # export collects. The session layer keeps the camera and the settings the renderer authors on it out of the baked
    # stage.
    camera_path = "/GenesisBakeCamera"
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Camera.Define(stage, camera_path)
        hydra_texture = omni.kit.hydra_texture.create_hydra_texture(
            name="usd_bake", width=16, height=16, usd_camera_path=camera_path
        )
        app.update()  # important
    logs.append(f"\tCreate render product: {time.time() - start_time}s.")

    # distill the material
    start_time = time.time()

    for usd_material_path in usd_material_paths:
        material_prim = stage.GetPrimAtPath(usd_material_path)
        distiller = omni.mdl.distill_and_bake.MdlDistillAndBake(material_prim, ouput_folder=output_dir)
        # The synchronous call returns before the material is baked, so the export below would collect the stage with
        # its material untouched. The result re-raises what the distill raised, and is False when the renderer holds no
        # such material: either way the export must never run on the unbaked stage.
        task = asyncio.ensure_future(distiller.distill_async())
        while not task.done():
            app.update()
        if not task.result():
            raise RuntimeError(f"Distilling material '{usd_material_path}' failed.")
        carb.log_info("Distilled: " + usd_material_path)
        logs.append(f"\tDistill: {time.time() - start_time}s, {usd_material_path}.")

    # export usd
    start_time = time.time()
    collector = omni.kit.usd.collect.Collector(stage.GetRootLayer().identifier, output_dir)
    task = asyncio.ensure_future(collector.collect())
    while not task.done():
        app.update()  # Otherwise it will be blocked by omni.kit.app.get_app().next_update_async()
    success, baked_file = task.result()

    logs.append(f"\tExport: {time.time() - start_time}s, {baked_file}.")
    print("Distill USD material:\n" + "\n".join(logs))

    # close omniverse app
    app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--usd_material_paths", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="warning")
    args = parser.parse_args()

    log_level = logging.getLevelName(
        min(max(logging.getLevelName(args.log_level.upper()), logging.INFO), logging.ERROR)
    ).lower()
    bake_usd_material(args.input_file, args.output_dir, args.usd_material_paths, args.device, log_level)
