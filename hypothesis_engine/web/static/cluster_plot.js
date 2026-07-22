(function () {
    const SVG_NS = "http://www.w3.org/2000/svg";
    const COLORS = [
        "#2864c5",
        "#2a8a4f",
        "#c08a00",
        "#b53636",
        "#6c4caf",
        "#00838f",
        "#8d5a00",
        "#4f6d7a",
        "#b24a72",
        "#5d7f28",
        "#7657b8",
        "#007057",
    ];
    const LIVE_REFRESH_EVENTS = new Set([
        "hypothesis_created",
        "review_completed",
        "match_complete",
        "tournament_match_complete",
        "session_done",
    ]);

    function colorFor(cluster) {
        const idx = Math.abs(Number(cluster) || 0) % COLORS.length;
        return COLORS[idx];
    }

    function svgEl(tag, attrs) {
        const el = document.createElementNS(SVG_NS, tag);
        Object.entries(attrs || {}).forEach(([key, value]) => {
            if (value !== null && value !== undefined) el.setAttribute(key, String(value));
        });
        return el;
    }

    function explicitLabel(point) {
        return typeof point.label === "string" && point.label.trim() ? point.label.trim() : "";
    }

    function pointTitle(point) {
        const parts = [];
        const label = explicitLabel(point);
        if (label) parts.push(label);
        else if (point.rank) parts.push("#" + point.rank);
        if (point.title) parts.push(point.title);
        if (point.elo !== null && point.elo !== undefined) parts.push("Elo " + Math.round(point.elo));
        if (point.source) parts.push(point.source);
        if (point.snippet) parts.push(point.snippet);
        return parts.join(" | ");
    }

    function boundsFor(points) {
        const xs = points.map((p) => Number(p.x)).filter(Number.isFinite);
        const ys = points.map((p) => Number(p.y)).filter(Number.isFinite);
        if (!xs.length || !ys.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
        let minX = Math.min(...xs);
        let maxX = Math.max(...xs);
        let minY = Math.min(...ys);
        let maxY = Math.max(...ys);
        if (minX === maxX) {
            minX -= 1;
            maxX += 1;
        }
        if (minY === maxY) {
            minY -= 1;
            maxY += 1;
        }
        return { minX, maxX, minY, maxY };
    }

    function scaler(points) {
        const width = 1000;
        const height = 620;
        const pad = { left: 64, right: 34, top: 36, bottom: 58 };
        const bounds = boundsFor(points);
        const xSpan = bounds.maxX - bounds.minX;
        const ySpan = bounds.maxY - bounds.minY;
        return {
            width,
            height,
            x: (value) => pad.left + ((Number(value) - bounds.minX) / xSpan) * (width - pad.left - pad.right),
            y: (value) => height - pad.bottom - ((Number(value) - bounds.minY) / ySpan) * (height - pad.top - pad.bottom),
        };
    }

    function formatMetric(value, digits) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
        return Number(value).toFixed(digits || 0);
    }

    function statusText(data) {
        const metrics = data.metrics || {};
        const parts = [];
        if (metrics.cluster_count !== undefined) parts.push(metrics.cluster_count + " clusters");
        if (data.view === "rag") {
            const sampled = metrics.sampled_chunks || metrics.points || (data.points || []).length;
            const total = metrics.kb_chunks || sampled;
            parts.push(sampled + "/" + total + " chunks");
            parts.push((metrics.top_label_count || 0) + " top hypotheses");
        } else {
            parts.push((metrics.points || (data.points || []).length) + " hypotheses");
            parts.push((metrics.top_label_count || 0) + " labels");
        }
        if (metrics.silhouette !== null && metrics.silhouette !== undefined) {
            parts.push("silhouette " + formatMetric(metrics.silhouette, 3));
        }
        return parts.join(" | ");
    }

    function renderAxes(svg, scale) {
        const axis = svgEl("g", { class: "cluster-axis" });
        axis.appendChild(svgEl("line", { x1: 64, y1: 562, x2: 966, y2: 562 }));
        axis.appendChild(svgEl("line", { x1: 64, y1: 36, x2: 64, y2: 562 }));
        const xLabel = svgEl("text", { x: 516, y: 604, "text-anchor": "middle" });
        xLabel.textContent = "PCA 1";
        const yLabel = svgEl("text", { x: 20, y: 306, transform: "rotate(-90 20 306)", "text-anchor": "middle" });
        yLabel.textContent = "PCA 2";
        axis.appendChild(xLabel);
        axis.appendChild(yLabel);
        svg.appendChild(axis);
    }

    function addPoint(group, point, scale, options) {
        if (!Number.isFinite(Number(point.x)) || !Number.isFinite(Number(point.y))) return;
        const x = scale.x(point.x);
        const y = scale.y(point.y);
        const color = colorFor(point.cluster);
        const r = options && options.overlay ? 7 : (point.label ? 5 : 3.5);
        const circle = svgEl("circle", {
            cx: x,
            cy: y,
            r,
            fill: color,
            stroke: options && options.overlay ? "#111827" : (point.label ? "#111827" : "none"),
            "stroke-width": options && options.overlay ? 2.25 : (point.label ? 1.5 : 0),
            opacity: options && options.overlay ? 1 : (point.kind === "kb_chunk" ? 0.58 : 0.72),
        });
        const title = svgEl("title");
        title.textContent = pointTitle(point);
        circle.appendChild(title);
        group.appendChild(circle);
    }

    function labelText(point, overlay) {
        const explicit = explicitLabel(point);
        if (explicit) return explicit;
        if (overlay && point.rank) return "H" + point.rank;
        const label = (point.rank ? "#" + point.rank + " " : "") + (point.title || point.id || "item");
        return label.length > 72 ? label.slice(0, 69) + "..." : label;
    }

    function addLabel(group, point, scale, index, overlay) {
        if (!Number.isFinite(Number(point.x)) || !Number.isFinite(Number(point.y))) return;
        const x = scale.x(point.x);
        const y = scale.y(point.y);
        const offsetY = overlay ? -14 : (index % 2 === 0 ? -11 : 17);
        const text = svgEl("text", {
            x: Math.min(Math.max(x + 9, 72), 940),
            y: Math.min(Math.max(y + offsetY, 48), 548),
            class: overlay ? "cluster-label cluster-label-overlay" : "cluster-label",
        });
        const title = svgEl("title");
        title.textContent = pointTitle(point);
        text.appendChild(title);
        text.appendChild(document.createTextNode(labelText(point, overlay)));
        group.appendChild(text);
    }

    function addLegend(svg, data) {
        const clusters = (data.clusters || []).slice(0, 12);
        if (!clusters.length) return;
        const legend = svgEl("g", { class: "cluster-legend" });
        clusters.forEach((cluster, idx) => {
            const x = 76 + (idx % 6) * 145;
            const y = 24 + Math.floor(idx / 6) * 20;
            legend.appendChild(svgEl("circle", { cx: x, cy: y, r: 5, fill: colorFor(cluster.cluster) }));
            const text = svgEl("text", { x: x + 10, y: y + 4 });
            const high = cluster.high_performing_hypotheses ? ", top " + cluster.high_performing_hypotheses : "";
            text.textContent = "C" + cluster.cluster + " n=" + cluster.size + high;
            legend.appendChild(text);
        });
        svg.appendChild(legend);
    }

    function renderPlot(root, data) {
        const status = root.querySelector("[data-cluster-status]");
        const stage = root.querySelector("[data-cluster-stage]");
        if (!stage || !status) return;
        stage.textContent = "";
        if (!data.available) {
            status.textContent = data.error || "Cluster plot is not available yet.";
            root.dataset.loaded = "false";
            return;
        }

        const points = Array.isArray(data.points) ? data.points : [];
        const overlays = Array.isArray(data.overlays) ? data.overlays : [];
        const allPoints = points.concat(overlays);
        if (!points.length) {
            status.textContent = "Cluster plot has no points yet.";
            root.dataset.loaded = "false";
            return;
        }

        status.textContent = statusText(data);
        const scale = scaler(allPoints);
        const svg = svgEl("svg", {
            class: "cluster-svg",
            viewBox: "0 0 " + scale.width + " " + scale.height,
            role: "img",
            "aria-label": root.dataset.clusterTitle || "Cluster plot",
        });
        const title = svgEl("title");
        title.textContent = root.dataset.clusterTitle || "Cluster plot";
        svg.appendChild(title);
        renderAxes(svg, scale);
        addLegend(svg, data);

        const background = svgEl("g", { class: "cluster-points" });
        points.forEach((point) => addPoint(background, point, scale));
        svg.appendChild(background);

        const labeled = svgEl("g", { class: "cluster-labels" });
        points.filter((point) => point.label).forEach((point, index) => {
            addPoint(labeled, point, scale);
            addLabel(labeled, point, scale, index, false);
        });
        overlays.forEach((point, index) => {
            addPoint(labeled, point, scale, { overlay: true });
            addLabel(labeled, point, scale, index, true);
        });
        svg.appendChild(labeled);
        stage.appendChild(svg);
        root.dataset.loaded = "true";
        root.dataset.stale = "false";
    }

    async function loadPlot(root, options) {
        if (!root || !root.dataset.clusterUrl) return;
        if (root.dataset.loading === "true") return;
        const status = root.querySelector("[data-cluster-status]");
        root.dataset.loading = "true";
        if (status && !(options && options.silent)) status.textContent = "Loading cluster plot...";
        try {
            const response = await fetch(root.dataset.clusterUrl, { headers: { Accept: "application/json" } });
            if (!response.ok) throw new Error("HTTP " + response.status);
            const data = await response.json();
            renderPlot(root, data);
        } catch (error) {
            if (status) status.textContent = "Cluster plot failed: " + error.message;
        } finally {
            root.dataset.loading = "false";
        }
    }

    function activePanel(container) {
        return container.querySelector("[data-tab-panel]:not([hidden])");
    }

    function loadPanelPlots(panel) {
        if (!panel) return;
        panel.querySelectorAll("[data-cluster-plot]").forEach((root) => {
            if (root.dataset.loaded !== "true" || root.dataset.stale === "true") loadPlot(root);
        });
    }

    function setActive(container, targetId) {
        container.querySelectorAll("[data-tab-target]").forEach((button) => {
            const active = button.dataset.tabTarget === targetId;
            button.setAttribute("aria-selected", active ? "true" : "false");
            button.classList.toggle("active", active);
        });
        container.querySelectorAll("[data-tab-panel]").forEach((panel) => {
            const active = panel.id === targetId;
            panel.hidden = !active;
            panel.classList.toggle("active", active);
            if (active) loadPanelPlots(panel);
        });
        window.dispatchEvent(new Event("resize"));
    }

    function initTabs(container) {
        const first = container.querySelector("[data-tab-target][aria-selected='true']") || container.querySelector("[data-tab-target]");
        container.querySelectorAll("[data-tab-target]").forEach((button) => {
            button.addEventListener("click", () => setActive(container, button.dataset.tabTarget));
        });
        if (first) setActive(container, first.dataset.tabTarget);
    }

    function initPlotControls() {
        document.querySelectorAll("[data-cluster-plot]").forEach((root) => {
            root.querySelectorAll("[data-cluster-refresh]").forEach((button) => {
                button.addEventListener("click", () => loadPlot(root));
            });
        });
    }

    function markPlotsStale() {
        document.querySelectorAll("[data-cluster-plot]").forEach((root) => {
            root.dataset.stale = "true";
            const panel = root.closest("[data-tab-panel]");
            if (panel && !panel.hidden && root.dataset.loaded === "true") {
                window.clearTimeout(root._clusterRefreshTimer);
                root._clusterRefreshTimer = window.setTimeout(() => loadPlot(root, { silent: true }), 1500);
            }
        });
    }

    function boot() {
        document.querySelectorAll("[data-session-tabs]").forEach(initTabs);
        initPlotControls();
        document.addEventListener("hypothesis-engine:event", (event) => {
            const name = event.detail && event.detail.name;
            const payload = event.detail && event.detail.payload ? event.detail.payload : {};
            if (LIVE_REFRESH_EVENTS.has(name) || (name === "task_completed" && payload.action === "RunProximityClustering")) {
                markPlotsStale();
            }
        });
        document.querySelectorAll("[data-session-tabs]").forEach((container) => loadPanelPlots(activePanel(container)));
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
