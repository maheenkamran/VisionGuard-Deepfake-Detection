(function () {
    var root = document.documentElement;
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /* 1. cursor spotlight, lit-up grid and parallax orbs */
    if (!reduce) {
        window.addEventListener("pointermove", function (e) {
            root.style.setProperty("--mx", e.clientX + "px");
            root.style.setProperty("--my", e.clientY + "px");
            var px = e.clientX / window.innerWidth - 0.5;
            var py = e.clientY / window.innerHeight - 0.5;
            document.querySelectorAll(".orb").forEach(function (o, i) {
                var d = (i + 1) * 30;
                o.style.translate = (px * d) + "px " + (py * d) + "px";
            });
        });
    }

    /* 2. cards: soft 3D tilt and a highlight that follows the cursor */
    document.querySelectorAll("[data-tilt], .card").forEach(function (card) {
        card.addEventListener("pointermove", function (e) {
            var r = card.getBoundingClientRect();
            var x = (e.clientX - r.left) / r.width;
            var y = (e.clientY - r.top) / r.height;
            card.style.setProperty("--gx", (x * 100) + "%");
            card.style.setProperty("--gy", (y * 100) + "%");
            if (!reduce && card.hasAttribute("data-tilt")) {
                card.style.transform = "perspective(900px) rotateX(" + ((0.5 - y) * 5) +
                    "deg) rotateY(" + ((x - 0.5) * 6) + "deg)";
            }
        });
        card.addEventListener("pointerleave", function () { card.style.transform = ""; });
    });

    /* 3. upload page: drag and drop, preview, scan animation */
    var input = document.getElementById("file");
    if (input) {
        var dz = document.getElementById("dropzone");
        var prev = document.getElementById("preview");
        var info = document.getElementById("file-info");
        var nameEl = document.getElementById("file-name");
        var warn = document.getElementById("file-warn");
        var form = document.getElementById("upload-form");
        var btn = document.getElementById("submit-btn");

        function show(file) {
            if (!file) { return; }
            prev.src = URL.createObjectURL(file);
            dz.classList.add("has");
            info.hidden = false;
            nameEl.textContent = file.name + " · " + (file.size / 1024 / 1024).toFixed(2) + " MB";
            // soft hint only: the server does the real validation, so submitting is never blocked
            var okType = /^image\/(jpeg|png|webp)$/.test(file.type);
            var okSize = file.size <= 5 * 1024 * 1024;
            warn.hidden = okType && okSize;
            warn.textContent = !okType ? "Unsupported type: the server will reject it"
                : (!okSize ? "Larger than 5 MB: the server will reject it" : "");
        }

        input.addEventListener("change", function () { show(input.files[0]); });
        ["dragenter", "dragover"].forEach(function (ev) {
            dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("drag"); });
        });
        ["dragleave", "drop"].forEach(function (ev) {
            dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("drag"); });
        });
        dz.addEventListener("drop", function (e) {
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                show(input.files[0]);
            }
        });

        form.addEventListener("submit", function () {
            dz.classList.add("scanning");
            btn.querySelector(".btn-label").textContent = "Scanning...";
            btn.disabled = true;
            // keep a small thumbnail in the browser only, so the result page can show it
            try {
                sessionStorage.removeItem("vg_preview");
                var c = document.createElement("canvas");
                var s = Math.min(1, 640 / Math.max(prev.naturalWidth, prev.naturalHeight));
                c.width = Math.max(1, Math.round(prev.naturalWidth * s));
                c.height = Math.max(1, Math.round(prev.naturalHeight * s));
                c.getContext("2d").drawImage(prev, 0, 0, c.width, c.height);
                sessionStorage.setItem("vg_preview", c.toDataURL("image/jpeg", 0.8));
            } catch (err) { /* preview is optional */ }
        });
    }

    /* 4. result page: preview image and animated confidence ring */
    var gauge = document.querySelector("[data-gauge]");
    if (gauge) {
        var conf = parseFloat(gauge.getAttribute("data-gauge")) || 0;
        var fg = gauge.querySelector(".ring-fg");
        var num = document.getElementById("gauge-num");
        var circ = 327;
        requestAnimationFrame(function () {
            fg.style.strokeDashoffset = circ - (circ * conf) / 100;
        });
        var start = null;
        function tick(t) {
            if (start === null) { start = t; }
            var p = Math.min(1, (t - start) / 1200);
            num.textContent = (conf * (1 - Math.pow(1 - p, 3))).toFixed(1);
            if (p < 1) { requestAnimationFrame(tick); }
        }
        if (!reduce) { requestAnimationFrame(tick); }

        var img = document.getElementById("result-img");
        var saved = null;
        try { saved = sessionStorage.getItem("vg_preview"); } catch (err) { saved = null; }
        if (saved && img) {
            img.src = saved;
            img.hidden = false;
            document.getElementById("img-fallback").hidden = true;
        }
    }

    /* 5. history page: bars, counters and live filter */
    document.querySelectorAll(".mini i").forEach(function (b) {
        requestAnimationFrame(function () { b.style.width = b.getAttribute("data-w") + "%"; });
    });
    var filter = document.getElementById("filter");
    if (filter) {
        var rows = document.querySelectorAll("#hist tbody tr");
        var none = document.getElementById("no-match");
        filter.addEventListener("input", function () {
            var q = filter.value.toLowerCase();
            var shown = 0;
            rows.forEach(function (r) {
                var hit = r.textContent.toLowerCase().indexOf(q) !== -1;
                r.hidden = !hit;
                if (hit) { shown += 1; }
            });
            none.hidden = shown !== 0;
        });
    }
})();