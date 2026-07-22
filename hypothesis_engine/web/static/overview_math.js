(function () {
    "use strict";

    var TEX_FRAGMENT = /\\(?:text|mathrm|mathbf|mathit|mathsf|mathcal|ce)\{[^{}]+\}(?:\s*(?:[_^]\s*(?:\{[^{}]+\}|[A-Za-z0-9+\-]+)))*|\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)(?:\s*(?:[_^]\s*(?:\{[^{}]+\}|[A-Za-z0-9+\-]+)))*/g;
    var SKIP_TAGS = new Set(["CODE", "PRE", "SCRIPT", "STYLE", "TEXTAREA", "KBD", "SAMP"]);

    function isAlreadyDelimited(text, start, end) {
        var before = text.slice(Math.max(0, start - 3), start);
        var after = text.slice(end, Math.min(text.length, end + 3));
        return before.endsWith("\\(") || before.endsWith("\\[") || before.endsWith("$") || after.startsWith("\\)") || after.startsWith("\\]") || after.startsWith("$");
    }

    function shouldSkip(node) {
        for (var el = node.parentElement; el; el = el.parentElement) {
            if (SKIP_TAGS.has(el.tagName)) {
                return true;
            }
        }
        return false;
    }

    function wrapBareTex(root) {
        var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        var nodes = [];
        var node;
        while ((node = walker.nextNode())) {
            if (!shouldSkip(node) && TEX_FRAGMENT.test(node.nodeValue || "")) {
                nodes.push(node);
            }
            TEX_FRAGMENT.lastIndex = 0;
        }

        nodes.forEach(function (textNode) {
            var text = textNode.nodeValue || "";
            var fragment = document.createDocumentFragment();
            var last = 0;
            var match;
            TEX_FRAGMENT.lastIndex = 0;
            while ((match = TEX_FRAGMENT.exec(text))) {
                var raw = match[0];
                var start = match.index;
                var end = start + raw.length;
                if (start > last) {
                    fragment.appendChild(document.createTextNode(text.slice(last, start)));
                }
                var replacement = isAlreadyDelimited(text, start, end) ? raw : "\\(" + raw + "\\)";
                fragment.appendChild(document.createTextNode(replacement));
                last = end;
            }
            if (last < text.length) {
                fragment.appendChild(document.createTextNode(text.slice(last)));
            }
            textNode.parentNode.replaceChild(fragment, textNode);
        });
    }

    window.MathJax = {
        tex: {
            inlineMath: [["\\(", "\\)"], ["$", "$"]],
            displayMath: [["\\[", "\\]"], ["$$", "$$"]],
            processEscapes: true,
            packages: {"[+]":["ams", "mhchem"]}
        },
        loader: {load: ["[tex]/ams", "[tex]/mhchem"]},
        options: {
            ignoreHtmlClass: "tex2jax_ignore",
            processHtmlClass: "overview-content|math-render-content"
        }
    };

    document.querySelectorAll("[data-render-math]").forEach(wrapBareTex);
}());
