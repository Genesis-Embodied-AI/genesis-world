"""Embedded browser UI for the simulation architecture inspector."""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Simulation Architecture Inspector</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071018;
      --panel: rgba(17, 31, 43, .92);
      --panel-2: #142737;
      --line: #294557;
      --text: #e5f2f7;
      --muted: #8ba5b3;
      --cyan: #56d9e9;
      --green: #77e1a6;
      --amber: #ffc56b;
      --violet: #b7a2ff;
      --danger: #ff7d8b;
      --shadow: 0 16px 50px rgba(0, 0, 0, .28);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% 0%, rgba(35, 112, 128, .22), transparent 30rem),
        radial-gradient(circle at 100% 25%, rgba(87, 62, 150, .16), transparent 34rem),
        var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      min-height: 100vh;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 16px clamp(16px, 4vw, 48px);
      border-bottom: 1px solid rgba(86, 217, 233, .14);
      background: rgba(7, 16, 24, .88);
      backdrop-filter: blur(16px);
    }
    .brand { margin-right: auto; }
    .brand h1 { margin: 0; font: 650 17px/1.25 system-ui, sans-serif; letter-spacing: .01em; }
    .brand p { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
    .controls { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    input[type="search"] {
      width: min(28vw, 300px);
      min-width: 170px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b1924;
      color: var(--text);
      padding: 9px 11px;
      outline: none;
    }
    input[type="search"]:focus { border-color: var(--cyan); box-shadow: 0 0 0 2px #56d9e922; }
    button {
      border: 1px solid #366476;
      border-radius: 8px;
      background: #123040;
      color: var(--text);
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: var(--cyan); }
    .live { display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12px; }
    .status { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); }
    .status.ok { background: var(--green); box-shadow: 0 0 10px var(--green); }
    .status.error { background: var(--danger); }
    main { width: min(1480px, 100%); margin: auto; padding: 26px clamp(14px, 4vw, 48px) 60px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(100px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }
    .stat, .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .stat { border-radius: 10px; padding: 12px 14px; }
    .stat strong { display: block; color: var(--cyan); font: 650 20px/1.2 system-ui, sans-serif; }
    .stat span { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .panel { border-radius: 12px; margin: 0 0 18px; overflow: hidden; }
    .panel-title {
      display: flex;
      align-items: baseline;
      gap: 10px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(20, 39, 55, .68);
    }
    .panel-title h2 { margin: 0; font: 650 14px/1.2 system-ui, sans-serif; }
    .panel-title span { color: var(--muted); font-size: 11px; }
    .graph-shell { position: relative; min-height: 420px; }
    #graph { height: min(68vh, 720px); min-height: 420px; background: #091721; }
    .graph-inspector {
      position: absolute;
      z-index: 4;
      top: 12px;
      right: 12px;
      width: min(390px, calc(100% - 24px));
      max-height: calc(100% - 24px);
      overflow: auto;
      border: 1px solid #3a697a;
      border-radius: 10px;
      background: rgba(9, 23, 33, .97);
      box-shadow: var(--shadow);
    }
    .graph-inspector[hidden] { display: none; }
    .inspector-header {
      position: sticky;
      top: 0;
      z-index: 1;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      background: #102632;
    }
    .inspector-header strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .inspector-header button { margin-left: auto; padding: 2px 8px; }
    .inspector-body { padding: 12px; }
    .graph-controls { margin-left: auto; display: flex; gap: 6px; }
    .graph-controls button { padding: 4px 9px; font-size: 11px; }
    .dependency-legend { display: flex; gap: 10px; margin-left: 8px; font-size: 10px; }
    .dependency-legend .strong { color: var(--amber); }
    .dependency-legend .weak { color: #669eff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 12px; padding: 14px; }
    details.card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #0d1d28;
    }
    details.card[open] { border-color: #3a697a; }
    summary {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 12px 13px;
      cursor: pointer;
      list-style: none;
    }
    summary::-webkit-details-marker { display: none; }
    summary::before { content: "›"; color: var(--cyan); font-size: 18px; transition: transform .15s; }
    details[open] > summary::before { transform: rotate(90deg); }
    .summary-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .summary-type { margin-left: auto; color: var(--muted); font-size: 10px; }
    .card-body { border-top: 1px solid var(--line); padding: 12px; }
    .section { margin-top: 13px; }
    .section:first-child { margin-top: 0; }
    .section h3 {
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .09em;
    }
    .chips { display: flex; gap: 5px; flex-wrap: wrap; }
    .chip {
      border: 1px solid #315062;
      border-radius: 999px;
      background: #112634;
      color: #c6d9e1;
      padding: 2px 7px;
      font-size: 10px;
    }
    .chip.graph { color: var(--violet); border-color: #554a82; }
    .chip.callback { color: var(--amber); border-color: #735d38; }
    .kv { display: grid; grid-template-columns: minmax(90px, auto) 1fr; gap: 4px 10px; font-size: 11px; }
    .kv dt { color: var(--muted); overflow-wrap: anywhere; }
    .kv dd { margin: 0; text-align: right; overflow-wrap: anywhere; }
    .data-node { border-left: 2px solid #285367; padding-left: 8px; margin: 9px 0; }
    .data-node > summary { padding: 3px 0; }
    .data-node > summary::before { font-size: 14px; }
    .data-content { padding-left: 9px; }
    .data-name { color: var(--green); }
    .buffer {
      display: grid;
      grid-template-columns: minmax(80px, 1fr) auto auto;
      gap: 8px;
      padding: 4px 0;
      border-bottom: 1px dotted #29414f;
      font-size: 10px;
    }
    .buffer span:nth-child(n+2) { color: var(--muted); }
    .empty { color: var(--muted); padding: 24px; text-align: center; }
    .error-box { color: var(--danger); white-space: pre-wrap; }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; gap: 10px; }
      .brand { margin: 0; }
      .controls { width: 100%; }
      input[type="search"] { flex: 1; width: auto; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .grid { grid-template-columns: minmax(0, 1fr); padding: 10px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Simulation Architecture Inspector</h1>
      <p id="engine-type">Waiting for a metadata snapshot…</p>
    </div>
    <div class="controls">
      <span id="status" class="status" title="Loading"></span>
      <input id="search" type="search" placeholder="Filter systems, buffers, actions…" autocomplete="off">
      <button id="refresh" type="button">Refresh</button>
      <label class="live"><input id="live" type="checkbox"> live</label>
    </div>
  </header>
  <main>
    <div id="stats" class="stats"></div>
    <section class="panel">
      <div class="panel-title"><h2>Engine metadata</h2><span>host configuration and device buffer shapes</span></div>
      <div id="engine" class="grid"></div>
    </section>
    <section class="panel">
      <div class="panel-title">
        <h2>System dependency graph</h2><span>drag nodes · wheel zoom · arrows point to dependencies</span>
        <div class="dependency-legend"><span class="strong">● require</span><span class="weak">● find</span></div>
        <div class="graph-controls">
          <button id="graph-zoom-in" type="button">+</button>
          <button id="graph-zoom-out" type="button">−</button>
          <button id="graph-fit" type="button">Fit</button>
          <button id="graph-reset" type="button">Reset</button>
          <button id="graph-unlock" type="button">Unlock</button>
        </div>
      </div>
      <div class="graph-shell">
        <div id="graph"><div class="empty">Loading graph…</div></div>
        <aside id="graph-inspector" class="graph-inspector" hidden></aside>
      </div>
    </section>
    <section class="panel">
      <div class="panel-title"><h2>Systems and data</h2><span id="systems-count"></span></div>
      <div id="systems" class="grid"></div>
    </section>
    <section class="panel">
      <div class="panel-title"><h2>Actions and collections</h2><span id="actions-count"></span></div>
      <div id="actions" class="grid"></div>
    </section>
    <section class="panel">
      <div class="panel-title"><h2>Pipelines and yield checkpoints</h2><span id="pipelines-count"></span></div>
      <div id="pipelines" class="grid"></div>
    </section>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
  <script>
    "use strict";
    let snapshot = null;
    let timer = null;
    let dependencyGraph = null;
    let selectedGraphNodeId = null;
    const graphPositions = new Map();
    const $ = (selector) => document.querySelector(selector);
    const shortType = (name) => (name || "unknown").split(".").pop();
    const text = (value) => value === null || value === undefined ? "—" :
      (typeof value === "object" ? JSON.stringify(value) : String(value));
    const el = (tag, className, content) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (content !== undefined) node.textContent = content;
      return node;
    };

    function section(title, content) {
      const node = el("div", "section");
      node.append(el("h3", "", title), content);
      return node;
    }

    function keyValues(values) {
      const list = el("dl", "kv");
      Object.entries(values || {}).forEach(([key, value]) => {
        list.append(el("dt", "", key), el("dd", "", text(value)));
      });
      return list;
    }

    function chips(values, className = "") {
      const list = el("div", "chips");
      values.forEach((value) => list.append(el("span", `chip ${className}`, value)));
      return list;
    }

    function renderBuffers(buffers) {
      const list = el("div");
      (buffers || []).forEach((buffer) => {
        const row = el("div", "buffer");
        row.append(
          el("span", "", buffer.name),
          el("span", "", `[${(buffer.shape || []).join(" × ")}]`),
          el("span", "", buffer.dtype || shortType(buffer.type))
        );
        list.append(row);
      });
      return list;
    }

    function renderData(data) {
      const node = el("details", "data-node");
      node.open = !data.ref;
      const label = data.ref ? `${data.name} → ${data.ref}` : `${data.name} · ${shortType(data.type)}`;
      node.append(el("summary", "data-name", label));
      const content = el("div", "data-content");
      if (data.config && Object.keys(data.config).length) content.append(keyValues(data.config));
      if ((data.declared_fields || []).length) {
        content.append(section(
          "Declared fields",
          chips(data.declared_fields.map(
            (field) => `${field.name}: ${shortType(field.type)}${field.initialized ? "" : " · uninitialized"}`
          ), "data")
        ));
      }
      content.append(renderBuffers(data.buffers));
      (data.children || []).forEach((child) => {
        if (child.system_ref) {
          content.append(el("div", "data-node", `${child.name} → ${child.system_ref}`));
        } else {
          content.append(renderData(child));
        }
      });
      node.append(content);
      return node;
    }

    function systemCard(system) {
      const details = el("details", "card");
      const summary = el("summary");
      summary.append(
        el("span", "summary-name", system.name),
        el("span", "summary-type", shortType(system.type))
      );
      const body = el("div", "card-body");
      const state = {
        id: system.id,
        valid: system.valid,
        building: system.building
      };
      body.append(section("State", keyValues(state)));
      if (Object.keys(system.config || {}).length) body.append(section("Configuration", keyValues(system.config)));
      if ((system.buffers || []).length) body.append(section("System buffers", renderBuffers(system.buffers)));
      if ((system.data || []).length) {
        const data = el("div");
        system.data.forEach((item) => data.append(renderData(item)));
        body.append(section("Data and buffers", data));
      }
      const refs = [
        ...(system.actions || []).map((item) => `action · ${item}`),
        ...(system.action_collections || []).map((item) => `collection · ${item}`),
        ...(system.pipelines || []).map((item) => `pipeline · ${item}`)
      ];
      if (refs.length) body.append(section("Runtime objects", chips(refs)));
      details.append(summary, body);
      return details;
    }

    function actionCard(action) {
      const details = el("details", "card");
      const summary = el("summary");
      summary.append(el("span", "summary-name", action.name), el("span", "summary-type", action.id));
      const body = el("div", "card-body");
      body.append(section("Action", keyValues({
        owner: action.owner,
        kernel: action.kernel,
        kind: action.kind,
        transientArity: action.transient_arity,
        data: (action.data_refs || action.data_types || []).map(
          (ref, index) => ref || action.data_types[index]
        ).join(", "),
        type: action.type
      })));
      details.append(summary, body);
      return details;
    }

    function collectionCard(collection) {
      const details = el("details", "card");
      const summary = el("summary");
      summary.append(el("span", "summary-name", collection.name), el("span", "summary-type", collection.id));
      const body = el("div", "card-body");
      body.append(section("Collection", keyValues({owner: collection.owner, type: collection.type})));
      body.append(section(
        "Ordered actions",
        collection.actions.length ? chips(collection.actions) : el("span", "empty", "No actions registered")
      ));
      details.append(summary, body);
      return details;
    }

    function pipelineCard(pipeline) {
      const details = el("details", "card");
      const summary = el("summary");
      summary.append(
        el("span", "summary-name", pipeline.name),
        el("span", "summary-type", pipeline.id)
      );
      const body = el("div", "card-body");
      body.append(section("Pipeline", keyValues({
        owner: pipeline.owner,
        type: pipeline.type,
        launched: pipeline.has_launched
      })));
      const stages = (pipeline.stages || []).map((stage, index) => `${index + 1} · ${stage.entry}`);
      if (stages.length) body.append(section("Stages", chips(stages, "graph")));
      const actionData = (pipeline.action_data_refs || pipeline.action_data_types || []).map(
        (ref, index) => ref || pipeline.action_data_types[index]
      );
      body.append(section(
        "Bound action data",
        actionData.length ? chips(actionData, "data") : el("span", "empty", "No action data bound")
      ));
      const callbacks = (pipeline.yield_callbacks || []).map(
        (item) => `${item.checkpoint} · ${item.callback}`
      );
      body.append(section(
        "Yield callbacks",
        callbacks.length ? chips(callbacks, "callback") : el("span", "empty", "No registered callbacks")
      ));
      details.append(summary, body);
      return details;
    }

    function graphNodeById(id) {
      if (id === "engine") {
        return {
          id: "engine",
          name: "SimEngine",
          type: snapshot.engine.type,
          config: snapshot.engine.config,
          buffers: snapshot.engine.buffers,
          data: [],
          actions: [],
          action_collections: [],
          pipelines: snapshot.pipelines.map((pipeline) => pipeline.id),
          valid: true,
          building: false
        };
      }
      return snapshot.systems.find((system) => system.id === id);
    }

    function openGraphInspector(id) {
      const inspector = $("#graph-inspector");
      const system = graphNodeById(id);
      if (!system) {
        inspector.hidden = true;
        return;
      }
      selectedGraphNodeId = id;
      const header = el("div", "inspector-header");
      header.append(
        el("strong", "", system.name),
        el("span", "summary-type", shortType(system.type))
      );
      const close = el("button", "", "×");
      close.type = "button";
      close.addEventListener("click", () => {
        selectedGraphNodeId = null;
        inspector.hidden = true;
        if (dependencyGraph) dependencyGraph.elements().removeClass("faded");
      });
      header.append(close);
      const body = el("div", "inspector-body");
      body.append(section("Metadata", keyValues({
        id: system.id,
        type: system.type,
        valid: system.valid,
        building: system.building
      })));
      if (Object.keys(system.config || {}).length) {
        body.append(section("Configuration", keyValues(system.config)));
      }
      if ((system.buffers || []).length) {
        body.append(section("Buffers", renderBuffers(system.buffers)));
      }
      if ((system.data || []).length) {
        const data = el("div");
        system.data.forEach((item) => data.append(renderData(item)));
        body.append(section("Data", data));
      }
      const outgoing = snapshot.dependencies
        .filter((edge) => edge.source === id)
        .map((edge) => `${edge.required ? "require" : "find"} → ${graphNodeById(edge.target)?.name || edge.target}`);
      const incoming = snapshot.dependencies
        .filter((edge) => edge.target === id)
        .map((edge) => `${graphNodeById(edge.source)?.name || edge.source} → ${edge.required ? "require" : "find"}`);
      body.append(section("Dependencies", outgoing.length ? chips(outgoing) : el("span", "empty", "No dependencies")));
      body.append(section("Dependents", incoming.length ? chips(incoming) : el("span", "empty", "No dependents")));
      const runtimeObjects = [
        ...(system.actions || []).map((item) => `action · ${item}`),
        ...(system.action_collections || []).map((item) => `collection · ${item}`),
        ...(system.pipelines || []).map((item) => `pipeline · ${item}`)
      ];
      if (runtimeObjects.length) body.append(section("Runtime objects", chips(runtimeObjects)));
      inspector.replaceChildren(header, body);
      inspector.hidden = false;
    }

    function renderGraph(systems, dependencies) {
      const host = $("#graph");
      if (!systems.length) {
        if (dependencyGraph) {
          dependencyGraph.destroy();
          dependencyGraph = null;
        }
        host.replaceChildren();
        host.append(el("div", "empty", "No systems registered"));
        return;
      }
      if (!window.cytoscape) {
        host.replaceChildren(el("div", "empty error-box", "Cytoscape failed to load"));
        return;
      }
      if (dependencyGraph) dependencyGraph.destroy();
      host.replaceChildren();
      const elements = [
        ...systems.map((system) => ({
          data: {
            id: system.id,
            label: system.name,
            type: shortType(system.type)
          },
          position: graphPositions.get(system.id)
        })),
        ...dependencies.map((edge, index) => ({
          data: {
            id: `dependency-${index}-${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            label: edge.attribute,
            required: edge.required ? 1 : 0
          }
        }))
      ];
      const haveSavedLayout = systems.every((system) => graphPositions.has(system.id));
      dependencyGraph = window.cytoscape({
        container: host,
        elements,
        layout: haveSavedLayout
          ? {name: "preset", fit: true, padding: 42}
          : {
              name: "grid",
              fit: true,
              padding: 42,
              cols: 3,
              avoidOverlap: true,
              avoidOverlapPadding: 36,
              condense: false
            },
        minZoom: 0.12,
        maxZoom: 4,
        wheelSensitivity: 0.18,
        boxSelectionEnabled: false,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "#122b39",
              "border-color": "#56d9e9",
              "border-width": 1.5,
              "color": "#e5f2f7",
              "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
              "font-size": 11,
              "label": "data(label)",
              "text-valign": "center",
              "text-halign": "center",
              "text-wrap": "wrap",
              "text-max-width": 135,
              "width": 160,
              "height": 58,
              "shape": "round-rectangle"
            }
          },
          {
            selector: 'node[id = "engine"]',
            style: {
              "background-color": "#433916",
              "border-color": "#ffc56b",
              "border-width": 2.5
            }
          },
          {
            selector: "node:selected",
            style: {
              "background-color": "#19475a",
              "border-color": "#ffc56b",
              "border-width": 3
            }
          },
          {
            selector: "node.faded, edge.faded",
            style: {"opacity": 0.12}
          },
          {
            selector: "edge",
            style: {
              "curve-style": "round-taxi",
              "taxi-direction": "auto",
              "taxi-turn": 34,
              "taxi-turn-min-distance": 12,
              "line-color": "#547486",
              "target-arrow-color": "#77e1a6",
              "target-arrow-shape": "triangle",
              "arrow-scale": 0.85,
              "width": 1.5,
              "label": ""
            }
          },
          {
            selector: "edge[required = 1]",
            style: {
              "line-color": "#ffc56b",
              "target-arrow-color": "#ffc56b",
              "width": 3
            }
          },
          {
            selector: "edge[required = 0]",
            style: {
              "line-color": "#669eff",
              "target-arrow-color": "#669eff",
              "width": 2
            }
          }
        ]
      });
      dependencyGraph.nodes().forEach((node) => {
        if (graphPositions.has(node.id())) node.lock();
      });
      dependencyGraph.on("dragfree", "node", (event) => {
        graphPositions.set(event.target.id(), event.target.position());
        event.target.lock();
      });
      dependencyGraph.on("tap", "node", (event) => {
        const neighborhood = event.target.closedNeighborhood();
        dependencyGraph.elements().addClass("faded");
        neighborhood.removeClass("faded");
        openGraphInspector(event.target.id());
      });
      dependencyGraph.on("tap", (event) => {
        if (event.target === dependencyGraph) {
          dependencyGraph.elements().removeClass("faded");
        }
      });
      if (selectedGraphNodeId && dependencyGraph.getElementById(selectedGraphNodeId).length) {
        const selected = dependencyGraph.getElementById(selectedGraphNodeId);
        dependencyGraph.elements().addClass("faded");
        selected.closedNeighborhood().removeClass("faded");
        openGraphInspector(selectedGraphNodeId);
      }
    }

    function render() {
      if (!snapshot) return;
      const query = $("#search").value.trim().toLowerCase();
      const includesQuery = (value) => !query || JSON.stringify(value).toLowerCase().includes(query);
      const systems = snapshot.systems.filter(includesQuery);
      const graphSystems = [
        {id: "engine", name: "SimEngine", type: snapshot.engine.type},
        ...systems
      ];
      const systemIds = new Set(graphSystems.map((system) => system.id));
      const dependencies = snapshot.dependencies.filter(
        (edge) => systemIds.has(edge.source) && systemIds.has(edge.target)
      );
      const pipelines = snapshot.pipelines.filter(includesQuery);
      const actions = snapshot.actions.filter(includesQuery);
      const collections = snapshot.action_collections.filter(includesQuery);
      $("#engine-type").textContent = snapshot.engine.type;
      $("#systems-count").textContent = `${systems.length} visible`;
      $("#actions-count").textContent = `${actions.length} actions · ${collections.length} collections`;
      $("#pipelines-count").textContent = `${pipelines.length} visible`;
      const callbacks = snapshot.pipelines.reduce(
        (count, pipeline) => count + pipeline.yield_callbacks.length, 0
      );
      const stats = [
        [snapshot.systems.length, "systems"],
        [snapshot.dependencies.length, "dependencies"],
        [snapshot.actions.length, "actions"],
        [snapshot.pipelines.length, "pipelines"],
        [callbacks, "callbacks"]
      ];
      $("#stats").replaceChildren(...stats.map(([value, label]) => {
        const card = el("div", "stat");
        card.append(el("strong", "", value), el("span", "", label));
        return card;
      }));
      const engineBody = el("div", "card-body");
      if (Object.keys(snapshot.engine.config || {}).length) {
        engineBody.append(section("Configuration", keyValues(snapshot.engine.config)));
      }
      if ((snapshot.engine.data || []).length) {
        const data = el("div");
        snapshot.engine.data.forEach((item) => data.append(renderData(item)));
        engineBody.append(section("Engine Data", data));
      }
      engineBody.append(section(
        "Buffers",
        snapshot.engine.buffers.length
          ? renderBuffers(snapshot.engine.buffers)
          : el("span", "empty", "No engine-level buffers")
      ));
      $("#engine").replaceChildren(engineBody);
      const systemsHost = $("#systems");
      systemsHost.replaceChildren(...systems.map(systemCard));
      if (!systems.length) systemsHost.append(el("div", "empty", "No matching systems"));
      const actionsHost = $("#actions");
      actionsHost.replaceChildren(
        ...actions.map(actionCard),
        ...collections.map(collectionCard)
      );
      if (!actions.length && !collections.length) {
        actionsHost.append(el("div", "empty", "No matching actions or collections"));
      }
      const pipelinesHost = $("#pipelines");
      pipelinesHost.replaceChildren(...pipelines.map(pipelineCard));
      if (!pipelines.length) pipelinesHost.append(el("div", "empty", "No matching pipelines"));
      renderGraph(graphSystems, dependencies);
    }

    async function refresh() {
      const status = $("#status");
      status.className = "status";
      status.title = "Refreshing";
      try {
        const response = await fetch("/api/snapshot", {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        snapshot = await response.json();
        status.className = "status ok";
        status.title = "Connected";
        render();
      } catch (error) {
        status.className = "status error";
        status.title = String(error);
        $("#graph").replaceChildren(el("div", "empty error-box", `Snapshot unavailable: ${error}`));
      }
    }

    $("#refresh").addEventListener("click", refresh);
    $("#search").addEventListener("input", render);
    $("#graph-zoom-in").addEventListener("click", () => {
      if (dependencyGraph) dependencyGraph.zoom({
        level: Math.min(dependencyGraph.maxZoom(), dependencyGraph.zoom() * 1.25),
        renderedPosition: {x: dependencyGraph.width() / 2, y: dependencyGraph.height() / 2}
      });
    });
    $("#graph-zoom-out").addEventListener("click", () => {
      if (dependencyGraph) dependencyGraph.zoom({
        level: Math.max(dependencyGraph.minZoom(), dependencyGraph.zoom() / 1.25),
        renderedPosition: {x: dependencyGraph.width() / 2, y: dependencyGraph.height() / 2}
      });
    });
    $("#graph-fit").addEventListener("click", () => {
      if (dependencyGraph) dependencyGraph.fit(undefined, 42);
    });
    $("#graph-unlock").addEventListener("click", () => {
      if (!dependencyGraph) return;
      dependencyGraph.nodes().unlock();
      dependencyGraph.elements().removeClass("faded");
    });
    $("#graph-reset").addEventListener("click", () => {
      graphPositions.clear();
      if (snapshot) render();
    });
    $("#live").addEventListener("change", (event) => {
      clearInterval(timer);
      timer = event.target.checked ? setInterval(refresh, 2000) : null;
    });
    refresh();
  </script>
</body>
</html>
"""

__all__ = ["INDEX_HTML"]
