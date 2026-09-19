# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Genesis は Python 製の汎用物理シミュレーションエンジン（`genesis-world` パッケージ、import 名は `genesis`）。ロボティクス／Embodied AI 向けに、複数の物理ソルバー（剛体、MPM、SPH、FEM、PBD、Stable Fluid）とそれらの結合（coupling）を単一フレームワークに統合している。計算バックエンドには [Taichi](https://github.com/taichi-dev/taichi) を使用し、CPU / Nvidia・AMD GPU / Apple Metal で動作する。

## セットアップとビルド

開発モードでのインストール（本リポジトリを編集する場合は必須）:
```bash
pip install -e ".[dev]"
```
PyTorch は事前に公式手順に従って別途インストールしておく必要がある。Python は `>=3.10,<3.13` が要件（`pyproject.toml` 参照）。

`genesis/ext/` 以下（LuisaRender, ParticleMesher）と `doc/` は git submodule。これらを含めて作業する場合は `git submodule update --init --recursive` が必要になる。

## よく使うコマンド

**フォーマット（black, line-length=120）:**
```bash
black --line-length 120 .
```
`pre-commit install` 済みであればコミット時に自動実行される（`.pre-commit-config.yaml`）。`genesis/ext/` は black のフォーマット対象外。

**テスト全体:**
```bash
pytest ./tests
```
CI (`.github/workflows/ci.yml`) は GPU コンテナ内で `pytest -v ./tests` と `pytest -v -m 'benchmarks' --backend gpu ./tests` を実行している。

**単一ファイル／単一テストの実行:**
```bash
pytest tests/test_rigid_physics.py
pytest tests/test_rigid_physics.py::test_function_name -v
```

**主なオプション（`tests/conftest.py` で定義、`pytest_addoption`）:**
- `--backend {cpu,gpu,...}`: シミュレーションバックエンドの指定（デフォルトは `cpu`）
- `--vis`: インタラクティブビューアを有効化（有効時は `pytest-xdist` の並列実行が自動的に無効化される）

**ベンチマークテストの除外／実行:**
デフォルトでは `pyproject.toml` の `addopts` により `-m "not benchmarks"` が付与され、`benchmarks` マーカー付きテストは除外される。ベンチマークだけを実行する場合は `-m benchmarks` を指定する（この場合も並列実行は自動的に無効化される）。

**キャッシュのクリーンアップ:**
```bash
gs clean   # genesis / taichi のキャッシュファイルを削除（`gs._main.main` 経由、pyproject.toml の [project.scripts] で定義）
```
`gs view <file>` (URDF/MJCF/メッシュのビューア) と `gs animate` サブコマンドも同じ CLI から利用可能。

## テストの前提知識

- 各テストは `initialize_genesis` フィクスチャ（`autouse=True`）により自動的に `gs.init()` → テスト本体 → `gs.destroy()` が呼ばれる。CPU バックエンドでは `precision="64"`, `debug=True` が強制される。
- `--backend` に GPU を指定してもマシンに GPU が無い場合、そのテストは自動的に `pytest.skip` される。
- `test_rigid_physics.py` は MuJoCo をリファレンス実装として使い、`mj_sim` / `gs_sim` フィクスチャで Genesis と MuJoCo のシミュレーション結果を突き合わせる構成になっている（`mpr_vanilla`, `adjacent_collision`, `multi_contact`, `dof_damping` などのマーカーで挙動を切り替え）。

## アーキテクチャ

### 初期化フロー
すべての利用は `gs.init(...)`（`genesis/__init__.py`）から始まる。ここで Taichi のバックエンド（arch）、精度（`ti_float`/`np_float`/`tc_float`、32/64bit）、乱数シード、デバッグモードなどグローバル状態を設定する。`gs.destroy()` で Taichi ランタイムをリセットし、生成済みの `Scene` を破棄する（`atexit` にも登録済み）。`init`/`destroy` は繰り返し呼び出し可能（テストが各テストごとに行う想定）。

### Scene → Simulator → Solvers という階層
- **`Scene`** (`genesis/engine/scene.py`): ユーザーが直接触るトップレベル API。エンティティの追加（`scene.add_entity(morph, material, surface, ...)`）、`scene.build()`（Taichi カーネルのコンパイルを含む）、`scene.step()` を提供する。
- **`Simulator`** (`genesis/engine/simulator.py`): シーンごとに1つ存在し、複数のソルバー（`RigidSolver`, `AvatarSolver`, `MPMSolver`, `SPHSolver`, `FEMSolver`, `SFSolver`, `PBDSolver`, `ToolSolver`）と `Coupler`（`genesis/engine/coupler.py`、ソルバー間の相互作用を扱う）をまとめて管理する。各ソルバーは対応する `*Options`（`genesis/options/solvers.py`）で設定される。
- **Entities** (`genesis/engine/entities/`): 各物理表現（`rigid_entity/`, `mpm_entity.py`, `sph_entity.py`, `fem_entity.py`, `pbd_entity.py`, `avatar_entity/`, `tool_entity/`, `drone_entity.py`, `hybrid_entity.py` など）に対応するエンティティクラス群。`Scene.add_entity` に渡す `morph`（形状・読み込み元。`genesis/options/morphs.py`）と `material`（物性。`genesis/engine/materials/`）の組み合わせでどのソルバー・エンティティが使われるか決まる。

### 剛体ソルバーは別格
`genesis/engine/solvers/rigid/` は最も規模が大きく複雑なサブシステムで、衝突検出（`collider_decomp.py`, `mpr_decomp.py`, `sdf_decomp.py`, `support_field_decomp.py`）、拘束ソルバー（`constraint_solver_decomp.py`, `constraint_solver_decomp_island.py`, `contact_island.py`）が分離されている。`rigid_solver_decomp.py` が中核。MJCF (`.xml`) / URDF / メッシュ (`.obj`, `.glb`, `.ply`, `.stl` など) の読み込みは `genesis/utils/mjcf.py`, `genesis/utils/urdf.py`, `genesis/utils/mesh.py` が担う。

### Taichi カーネルの扱い
物理演算の内側のループは Taichi の `@ti.data_oriented` クラス／`@ti.kernel`／`@ti.func` で書かれている（例: `Simulator` 自体も `@ti.data_oriented`）。これらのコードは通常の Python セマンティクスに従わない（型付き、静的コンパイルされる）ため、Taichi 特有の制約（動的な分岐・可変長データ構造の制限、スカラー化など）を踏まえて編集する必要がある。`genesis/constants.py` の `GS_ARCH` / `TI_ARCH` がプラットフォームとバックエンドの対応表。

### 微分可能性
`genesis/grad/`（`creation_ops.py`, `tensor.py`）が自動微分対応のテンソル生成を扱う。現状 MPM ソルバーと ToolSolver が微分可能性をサポートしており、他のソルバー（特に剛体・関節ソルバー）は今後対応予定。

### レンダリング／可視化
`genesis/vis/` にラスタライザ（`rasterizer.py`, `rasterizer_context.py`）とレイトレーサー（`raytracer.py`、LuisaRender submodule 利用）、インタラクティブビューア（`viewer.py`）、カメラ（`camera.py`）がある。フォトリアルなレイトレースレンダリングを使う場合は `genesis/ext/LuisaRender` submodule のビルドが前提（Docker イメージにはビルド済みで含まれる）。

### `genesis/ext/`
サードパーティ／フォーク由来のコードを格納するディレクトリ（`pyrender`, `urdfpy`, `isaacgym` 由来コードなど）。black のフォーマット対象外であり、基本的に外部プロジェクトのコードとして扱い、独自の変更は最小限にとどめる。

## 参考ドキュメント

- 多言語 README: `README.md`（英語）, `README_JA.md`, `README_CN.md`, `README_FR.md`, `README_KR.md`
- 詳細なユーザーガイド・API リファレンスは https://genesis-world.readthedocs.io にある（`doc/` submodule のソースから生成）
- コントリビューションガイド: `.github/CONTRIBUTING.md`（PR タイトルに `[BUG FIX]` / `[FEATURE]` / `[MISC]` のいずれかを付ける、`pre-commit` の利用、など）
