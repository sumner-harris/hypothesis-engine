(function () {
    const STAGES = ["generation", "literature_review", "reflection", "ranking", "evolution", "proximity", "metareview"];
    const EVENT_NAMES = [
        "session_started",
        "task_started",
        "task_completed",
        "task_failed",
        "task_warning",
        "hypothesis_created",
        "review_completed",
        "match_complete",
        "tournament_match_complete",
        "session_done",
        "human_feedback",
        "session_paused",
        "session_resumed",
        "session_aborted",
    ];

    function parseState(root) {
        try {
            return JSON.parse(root.dataset.state || "{}");
        } catch {
            return {};
        }
    }

    function eventPayload(data) {
        if (data && typeof data === "object" && Object.prototype.hasOwnProperty.call(data, "payload")) {
            return data.payload || {};
        }
        return data || {};
    }

    function stageForTask(agent, action) {
        if (STAGES.includes(agent)) return agent;
        if (action === "GenerateSystemFeedback" || action === "GenerateFinalResearchOverview") {
            return "metareview";
        }
        return null;
    }

    function eventLine(name, data) {
        const payload = eventPayload(data);
        const stamp = new Date().toISOString().slice(11, 19);
        if (name === "task_warning") {
            return stamp + " task_warning " + [payload.agent, payload.action, payload.reason]
                .filter(Boolean)
                .join(" · ");
        }
        if (name === "task_failed") {
            return stamp + " task_failed " + [payload.agent, payload.action, payload.err]
                .filter(Boolean)
                .join(" · ");
        }
        if (name === "task_completed" && payload.kind === "noop" && payload.extra && payload.extra.reason) {
            return stamp + " task_completed noop " + [payload.agent, payload.action, payload.extra.reason]
                .filter(Boolean)
                .join(" · ");
        }
        if (name === "task_completed" && payload.extra && payload.extra.recovered_record_review) {
            return stamp + " task_completed recovered_review " + [payload.agent, payload.action, payload.extra.verdict]
                .filter(Boolean)
                .join(" · ");
        }
        return stamp + " " + name + " " + (payload ? JSON.stringify(payload).slice(0, 220) : "");
    }

    function pushLog(log, name, data) {
        if (!log) return;
        const li = document.createElement("li");
        li.textContent = eventLine(name, data);
        log.insertBefore(li, log.firstChild);
        while (log.children.length > 80) log.removeChild(log.lastChild);
    }

    function init(root) {
        const state = parseState(root);
        const counts = {};
        const taskStages = new Map();
        const taskOrigins = new Map();
        const litReviewSources = new Set(state.active_literature_review_sources || []);
        STAGES.forEach((stage) => {
            counts[stage] = Number((state.counts || {})[stage] || 0);
        });
        (state.active_tasks || []).forEach((task) => {
            if (task.task_id && task.stage) {
                taskStages.set(task.task_id, task.stage);
                if (task.target_created_by) taskOrigins.set(task.task_id, task.target_created_by);
            }
        });

        const logId = root.dataset.logId || "events-log";
        const log = document.getElementById(logId);

        function nodeBox(stage) {
            const node = root.querySelector('[data-workflow-stage="' + stage + '"]');
            if (!node) return null;
            const rootRect = root.getBoundingClientRect();
            const rect = node.getBoundingClientRect();
            return {
                left: rect.left - rootRect.left,
                right: rect.right - rootRect.left,
                top: rect.top - rootRect.top,
                bottom: rect.bottom - rootRect.top,
                width: rect.width,
                height: rect.height,
            };
        }

        function anchorPoint(box, side, fraction) {
            const out = 20;
            const f = fraction == null ? 0.5 : fraction;
            if (side === "top") return { x: box.left + box.width * f, y: box.top - out };
            if (side === "bottom") return { x: box.left + box.width * f, y: box.bottom + out };
            if (side === "left") return { x: box.left - out, y: box.top + box.height * f };
            if (side === "right") return { x: box.right + out, y: box.top + box.height * f };
            return { x: box.left + box.width / 2, y: box.top + box.height / 2 };
        }

        function pathLine(start, end) {
            return "M " + start.x + " " + start.y + " L " + end.x + " " + end.y;
        }

        function pathCurve(start, c1, c2, end) {
            return "M " + start.x + " " + start.y + " C " + c1.x + " " + c1.y + ", " + c2.x + " " + c2.y + ", " + end.x + " " + end.y;
        }

        function pathForLink(from, to) {
            const source = nodeBox(from);
            const target = nodeBox(to);
            if (!source || !target) return null;

            if (from === "supervisor") {
                const sourceSlot = to === "generation" ? 0.25 : (to === "ranking" ? 0.75 : 0.5);
                return pathLine(anchorPoint(source, "bottom", sourceSlot), anchorPoint(target, "top", 0.5));
            }
            if (from === "generation" && to === "literature_review") {
                return pathLine(anchorPoint(source, "bottom", 0.5), anchorPoint(target, "top", 0.5));
            }
            if (from === "reflection" && to === "literature_review") {
                const start = anchorPoint(source, "bottom", 0.22);
                const end = anchorPoint(target, "right", 0.38);
                const bendY = (source.bottom + target.top) / 2;
                return pathCurve(
                    start,
                    { x: start.x - 24, y: bendY },
                    { x: end.x + 84, y: bendY },
                    end
                );
            }
            if (from === "evolution" && to === "literature_review") {
                return pathLine(anchorPoint(source, "left", 0.5), anchorPoint(target, "right", 0.5));
            }
            if (from === "generation" && to === "reflection") {
                return pathLine(anchorPoint(source, "right", 0.5), anchorPoint(target, "left", 0.5));
            }
            if (from === "reflection" && to === "ranking") {
                return pathLine(anchorPoint(source, "right", 0.5), anchorPoint(target, "left", 0.5));
            }
            if (from === "ranking" && to === "proximity") {
                return pathLine(anchorPoint(source, "right", 0.5), anchorPoint(target, "left", 0.5));
            }
            if (from === "proximity" && to === "ranking") {
                const start = anchorPoint(source, "bottom", 0.35);
                const end = anchorPoint(target, "bottom", 0.68);
                const bendY = Math.max(source.bottom, target.bottom) + 54;
                return pathCurve(
                    start,
                    { x: start.x - 56, y: bendY },
                    { x: end.x + 56, y: bendY },
                    end
                );
            }
            if (from === "ranking" && to === "evolution") {
                const start = anchorPoint(source, "bottom", 0.18);
                const end = anchorPoint(target, "top", 0.76);
                const bendY = (source.bottom + target.top) / 2;
                return pathCurve(
                    start,
                    { x: start.x - 72, y: bendY },
                    { x: end.x + 140, y: bendY },
                    end
                );
            }
            if (from === "metareview" && to === "evolution") {
                return pathLine(anchorPoint(source, "left", 0.5), anchorPoint(target, "right", 0.5));
            }
            if (from === "evolution" && to === "reflection") {
                const start = anchorPoint(source, "top", 0.62);
                const end = anchorPoint(target, "bottom", 0.22);
                const bendY = (source.top + target.bottom) / 2;
                return pathCurve(
                    start,
                    { x: start.x + 18, y: bendY },
                    { x: end.x - 112, y: bendY },
                    end
                );
            }
            return pathLine(anchorPoint(source, "right", 0.5), anchorPoint(target, "left", 0.5));
        }

        function clamp(value, min, max) {
            return Math.min(Math.max(value, min), max);
        }

        function setLabelPosition(name, x, y, rect) {
            const label = root.querySelector('[data-workflow-label="' + name + '"]');
            if (!label) return;
            label.style.left = "0px";
            label.style.right = "auto";
            label.style.top = "0px";
            label.style.bottom = "auto";
            const labelRect = label.getBoundingClientRect();
            const left = clamp(x - labelRect.width / 2, 10, Math.max(10, rect.width - labelRect.width - 10));
            const top = clamp(y - labelRect.height / 2, 10, Math.max(10, rect.height - labelRect.height - 10));
            label.style.left = left + "px";
            label.style.top = top + "px";
        }

        function layoutLabels(rect) {
            const generation = nodeBox("generation");
            const literatureReview = nodeBox("literature_review");
            const reflection = nodeBox("reflection");
            const ranking = nodeBox("ranking");
            const evolution = nodeBox("evolution");
            const proximity = nodeBox("proximity");
            const metareview = nodeBox("metareview");
            if (!generation || !literatureReview || !reflection || !ranking || !evolution || !proximity || !metareview) return;

            setLabelPosition(
                "hyp",
                (generation.right + reflection.left) / 2,
                generation.top + generation.height * 0.5 - 22,
                rect
            );
            setLabelPosition(
                "selection",
                literatureReview.right + 42,
                (generation.bottom + literatureReview.top) / 2,
                rect
            );
            setLabelPosition(
                "review",
                (reflection.right + ranking.left) / 2,
                reflection.top + reflection.height * 0.5 - 22,
                rect
            );
            setLabelPosition(
                "elo",
                ranking.right + 28,
                (ranking.bottom + proximity.top) / 2,
                rect
            );
            setLabelPosition(
                "feedback",
                (evolution.right + metareview.left) / 2,
                metareview.top + metareview.height * 0.5 - 22,
                rect
            );
            setLabelPosition(
                "reentry",
                (evolution.right + reflection.left) / 2,
                (evolution.top + reflection.bottom) / 2 + 28,
                rect
            );
        }

        function hasActiveReflectionFrom(origin) {
            for (const [taskId, stage] of taskStages.entries()) {
                if (stage === "reflection" && taskOrigins.get(taskId) === origin) return true;
            }
            return false;
        }

        function layoutLinks() {
            const svg = root.querySelector(".workflow-svg");
            if (!svg) return;
            const rect = root.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            svg.setAttribute("viewBox", "0 0 " + rect.width + " " + rect.height);
            root.querySelectorAll("[data-workflow-link-to]").forEach((link) => {
                const from = link.dataset.workflowLinkFrom;
                const to = link.dataset.workflowLinkTo;
                if (!from || !to) return;
                const d = pathForLink(from, to);
                if (d) link.setAttribute("d", d);
            });
            layoutLabels(rect);
        }

        function render() {
            STAGES.forEach((stage) => {
                const count = counts[stage] || 0;
                const active = count > 0;
                root.querySelectorAll('[data-workflow-stage="' + stage + '"]').forEach((node) => {
                    node.classList.toggle("active", active);
                    node.setAttribute("aria-current", active ? "step" : "false");
                });
                root.querySelectorAll('[data-workflow-count-for="' + stage + '"]').forEach((el) => {
                    el.textContent = active ? String(count) + " active" : "Idle";
                });
            });
            root.querySelectorAll("[data-workflow-link-to]").forEach((link) => {
                const from = link.dataset.workflowLinkFrom;
                const to = link.dataset.workflowLinkTo;
                const mode = link.dataset.workflowLinkActive || "target";
                const fromActive = (counts[from] || 0) > 0;
                const toActive = (counts[to] || 0) > 0;
                let active = toActive;
                if (mode === "source") active = fromActive;
                else if (mode === "either") active = fromActive || toActive;
                else if (mode === "both") active = fromActive && toActive;
                else if (mode === "reflection-generation") active = hasActiveReflectionFrom("generation");
                else if (mode === "reflection-evolution") active = hasActiveReflectionFrom("evolution");
                else if (mode === "literature-review-source") active = to === "literature_review" && litReviewSources.has(from);
                link.classList.toggle("active", active);
            });
            layoutLinks();
        }

        function increment(stage) {
            if (!stage) return;
            counts[stage] = (counts[stage] || 0) + 1;
            render();
        }

        function decrement(stage) {
            if (!stage) return;
            counts[stage] = Math.max(0, (counts[stage] || 0) - 1);
            render();
        }

        async function sync() {
            if (!root.dataset.workflowUrl) return;
            try {
                const response = await fetch(root.dataset.workflowUrl, { headers: { Accept: "application/json" } });
                if (!response.ok) return;
                const next = await response.json();
                STAGES.forEach((stage) => {
                    counts[stage] = Number((next.counts || {})[stage] || 0);
                });
                taskStages.clear();
                taskOrigins.clear();
                litReviewSources.clear();
                (next.active_literature_review_sources || []).forEach((source) => {
                    litReviewSources.add(source);
                });
                (next.active_tasks || []).forEach((task) => {
                    if (task.task_id && task.stage) {
                        taskStages.set(task.task_id, task.stage);
                        if (task.target_created_by) taskOrigins.set(task.task_id, task.target_created_by);
                    }
                });
                render();
            } catch {
                return;
            }
        }

        function applyEvent(name, data) {
            const payload = eventPayload(data);
            if (name === "task_started") {
                const stage = stageForTask(payload.agent, payload.action);
                if (payload.task_id && stage && !taskStages.has(payload.task_id)) {
                    taskStages.set(payload.task_id, stage);
                    if (payload.target_created_by) taskOrigins.set(payload.task_id, payload.target_created_by);
                    increment(stage);
                    if (stage === "reflection") sync();
                }
            } else if (name === "task_completed" || name === "task_failed") {
                const taskId = payload.task_id;
                const stage = taskStages.get(taskId);
                if (stage) {
                    taskStages.delete(taskId);
                    taskOrigins.delete(taskId);
                    decrement(stage);
                } else {
                    sync();
                }
            } else if (name === "session_done" || name === "session_aborted") {
                STAGES.forEach((stage) => {
                    counts[stage] = 0;
                });
                taskStages.clear();
                taskOrigins.clear();
                litReviewSources.clear();
                render();
            }
        }

        function handle(name, rawData) {
            let data = rawData;
            if (typeof rawData === "string") {
                try {
                    data = JSON.parse(rawData);
                } catch {
                    data = { payload: rawData };
                }
            }
            pushLog(log, name, data);
            applyEvent(name, data);
            document.dispatchEvent(
                new CustomEvent("hypothesis-engine:event", { detail: { name: name, data: data, payload: eventPayload(data) } })
            );
        }

        render();
        if (window.ResizeObserver) {
            const resizeObserver = new ResizeObserver(() => layoutLinks());
            resizeObserver.observe(root);
            root._workflowResizeObserver = resizeObserver;
        } else {
            window.addEventListener("resize", layoutLinks);
        }
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(layoutLinks).catch(() => {});
        }
        const syncTimer = window.setInterval(sync, 5000);
        root._workflowSyncTimer = syncTimer;
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) sync();
        });
        if (!root.dataset.eventsUrl) return;
        const es = new EventSource(root.dataset.eventsUrl);
        EVENT_NAMES.forEach((name) => {
            es.addEventListener(name, (event) => handle(name, event.data));
        });
        es.onerror = () => {
            pushLog(log, "[sse error]", null);
            sync();
        };
    }

    function boot() {
        document.querySelectorAll("[data-workflow-diagram]").forEach(init);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
