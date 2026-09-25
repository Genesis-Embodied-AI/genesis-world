import argparse
import asyncio
import importlib.metadata
import logging
import os
import time

# Kit 106 is the last release with wheels for Python 3.10 and 3.11; Python 3.12 gets Kit 107 and later, whose extension
# registry, viewport and distiller differ from Kit 106 (see 'omni_bootstrap' and 'bake_usd_material').
KIT_MAJOR = int(importlib.metadata.version("omniverse-kit").split(".")[0])


def omni_bootstrap(device=0, log_level="warning"):
    import omni.kit_app

    app = omni.kit_app.KitApp()
    kit_dir = os.path.dirname(os.path.abspath(os.path.realpath(omni.kit_app.__file__)))
    kit_path = os.path.join(kit_dir, "apps", "omni.app.empty.kit")

    omni_extensions = ["omni.usd", "omni.kit.material.library", "omni.kit.usd.collect", "omni.mdl.distill_and_bake"]
    # The distiller reads the material from the RTX render engine attached to the stage, which the viewport window
    # attaches and feeds when the stage opens. Kit 106 reaches both through the replicator, whose dependencies resolve
    # from the Kit 107 shared registry; Kit 107 and later ship them in their own registry.
    if KIT_MAJOR < 107:
        omni_extensions.append("omni.replicator.core")
    else:
        omni_extensions += ["omni.hydra.rtx", "omni.kit.viewport.window"]
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
        "--no-window",
        "--portable",  # TODO: set portable root to avoid extension conflicts
    ]
    if KIT_MAJOR < 107:
        app_args += [
            "--/exts/omni.kit.registry.nucleus/registries/0/name=kit/default",
            "--/exts/omni.kit.registry.nucleus/registries/0/url=https://ovextensionsprod.blob.core.windows.net/exts/kit/prod/107/shared",
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
    import omni.kit.usd.collect

    # open stage
    start_time = time.time()
    omni.usd.get_context().open_stage(input_file)
    logs.append(f"\tOpen stage: {time.time() - start_time}s.")

    # create render product
    start_time = time.time()
    stage = omni.usd.get_context().get_stage()
    app.update()  # important
    logs.append(f"\tCreate render product: {time.time() - start_time}s.")

    # distill the material
    start_time = time.time()

    for usd_material_path in usd_material_paths:
        material_prim = stage.GetPrimAtPath(usd_material_path)
        distiller = omni.mdl.distill_and_bake.MdlDistillAndBake(material_prim, ouput_folder=output_dir)
        if KIT_MAJOR < 107:
            distiller.distill()
        else:
            # The synchronous call of Kit 107 and later returns before the material is baked, so the export below would
            # collect the stage with its material untouched. The result re-raises what the distill raised, and is False
            # when the renderer holds no such material; either way the export must never run on the unbaked stage.
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
