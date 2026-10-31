/**
 * CAVShield Real-Time Security Operations Center (SOC) Console
 * Frontend Logic (Vanilla JS, Zero Frameworks, Offline Plotly Integration)
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Attack Scenario Definitions
  // -------------------------------------------------------------------------
  const ATTACKS = [
    { name: "spoof_brake", label: "Spoof Brake Command (0x200)" },
    { name: "spoof_accel", label: "Spoof Accel Command (0x201)" },
    { name: "spoof_speed", label: "Spoof Vehicle Speed (0x101)" },
    { name: "spoof_rpm", label: "Spoof Engine RPM (0x100)" },
    { name: "spoof_gear", label: "Spoof Reverse Gear (0x104)" },
    { name: "flood", label: "CAN Bus Flooding (700 Hz)" },
    { name: "fuzz", label: "CAN Bus Fuzzing (300 Hz)" },
    { name: "replay", label: "Historical CAN Replay" },
    { name: "v2x_phantom", label: "V2X Phantom Hazard (Ghost)" },
    { name: "v2x_fake_ebrake", label: "V2X Fake Emergency Brake" },
    { name: "v2x_gps", label: "V2X Location Drift" },
  ];

  // -------------------------------------------------------------------------
  // Application State Cache (Used for Selective DOM Mutation)
  // -------------------------------------------------------------------------
  const state = {
    protection: true,
    activeAttacks: new Set(),
    demoRunning: false,
    lastJson: {
      banner: "",
      twin: "",
      threatKpis: "",
      failsafe: "",
      incidents: "",
      can: "",
      v2x: "",
      securityEvents: "",
      metrics: "",
      demo: "",
    },
    canFilter: "all",
    v2xFilter: "all",
    lastSnapshot: null,
    lastChartRenderTime: 0,
    plotlyInitialized: false,
  };

  // -------------------------------------------------------------------------
  // Helper: Secure DOM Text Construction (Defends against XSS)
  // -------------------------------------------------------------------------
  function createTextEl(tagName, textContent, className = "") {
    const el = document.createElement(tagName);
    if (className) el.className = className;
    el.textContent = textContent == null ? "" : String(textContent);
    return el;
  }

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // -------------------------------------------------------------------------
  // DOM References
  // -------------------------------------------------------------------------
  const dom = {
    // Header & Mode Banner
    chipBus: document.getElementById("chipBus"),
    chipAi: document.getElementById("chipAi"),
    chipConn: document.getElementById("chipConn"),
    connStatusText: document.getElementById("connStatusText"),
    modeBanner: document.getElementById("modeBanner"),
    modePill: document.getElementById("modePill"),
    bannerHeadline: document.getElementById("bannerHeadline"),
    bannerAction: document.getElementById("bannerAction"),
    protBadge: document.getElementById("protBadge"),

    // Demo Callout Bar
    demoCaptionBar: document.getElementById("demoCaptionBar"),
    demoCaptionText: document.getElementById("demoCaptionText"),

    // Left Controls
    attackCountBadge: document.getElementById("activeAttackCount"),
    attackBtnGroup: document.getElementById("attackBtnGroup"),
    btnRealObstacle: document.getElementById("btnRealObstacle"),
    btnProtectionToggle: document.getElementById("btnProtectionToggle"),
    btnProtectionStatusText: document.getElementById("btnProtectionStatusText"),
    btnGuidedDemo: document.getElementById("btnGuidedDemo"),
    btnResetSystem: document.getElementById("btnResetSystem"),

    // Digital Twin Cards
    twinSpeed: document.getElementById("twinSpeed"),
    twinSpeedSub: document.getElementById("twinSpeedSub"),
    twinRpm: document.getElementById("twinRpm"),
    twinRpmSub: document.getElementById("twinRpmSub"),
    twinDecel: document.getElementById("twinDecel"),
    twinBrakeSub: document.getElementById("twinBrakeSub"),
    twinSteering: document.getElementById("twinSteering"),
    twinSteeringSub: document.getElementById("twinSteeringSub"),
    twinGear: document.getElementById("twinGear"),
    twinGearSub: document.getElementById("twinGearSub"),
    twinSafetyCard: document.getElementById("twinSafetyCard"),
    twinSafetyVal: document.getElementById("twinSafetyVal"),
    twinSafetySub: document.getElementById("twinSafetySub"),

    // Threat Monitoring & KPIs
    kpiAttackType: document.getElementById("kpiAttackType"),
    kpiConfidence: document.getElementById("kpiConfidence"),
    kpiRiskLevel: document.getElementById("kpiRiskLevel"),
    explanationBox: document.getElementById("explanationBox"),
    explanationTitle: document.getElementById("explanationTitle"),
    explanationBody: document.getElementById("explanationBody"),

    // Fail-Safe Status
    failsafeIncidentCount: document.getElementById("failsafeIncidentCount"),
    statRejectedCommands: document.getElementById("statRejectedCommands"),
    statRateLimited: document.getElementById("statRateLimited"),
    statQuarantined: document.getElementById("statQuarantined"),
    statRejectedV2x: document.getElementById("statRejectedV2x"),
    actionsList: document.getElementById("actionsList"),
    incidentsTbody: document.querySelector("#incidentsTable tbody"),

    // Streams & Filters
    canTbody: document.querySelector("#canTable tbody"),
    v2xTbody: document.querySelector("#v2xTable tbody"),
    canFlaggedBadge: document.getElementById("canFlaggedBadge"),
    v2xFlaggedBadge: document.getElementById("v2xFlaggedBadge"),
    canFilterAll: document.getElementById("canFilterAll"),
    canFilterFlagged: document.getElementById("canFilterFlagged"),
    v2xFilterAll: document.getElementById("v2xFilterAll"),
    v2xFilterFlagged: document.getElementById("v2xFilterFlagged"),

    // Security Events Feed
    secEventsCount: document.getElementById("secEventsCount"),
    securityEventsTbody: document.querySelector("#securityEventsTable tbody"),

    // Metrics Strip
    metricAcc: document.getElementById("metricAcc"),
    metricPrec: document.getElementById("metricPrec"),
    metricRec: document.getElementById("metricRec"),
    metricF1: document.getElementById("metricF1"),
    metricFpr: document.getElementById("metricFpr"),
    metricDetLat: document.getElementById("metricDetLat"),
    metricDetLatSub: document.getElementById("metricDetLatSub"),
    metricFsLat: document.getElementById("metricFsLat"),
    metricFsLatSub: document.getElementById("metricFsLatSub"),
    metricInfLat: document.getElementById("metricInfLat"),
    metricInfLatSub: document.getElementById("metricInfLatSub"),
  };

  // -------------------------------------------------------------------------
  // REST API Client
  // -------------------------------------------------------------------------
  const api = {
    async post(endpoint, data = {}) {
      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          console.error(`API Error [${endpoint}]:`, err.detail || res.statusText);
        }
        return res.ok;
      } catch (err) {
        console.error(`Network Error [${endpoint}]:`, err);
        return false;
      }
    },

    startAttack(name, duration = null) {
      return this.post(`/api/attack/${encodeURIComponent(name)}/start`, { duration });
    },

    stopAttack(name) {
      return this.post(`/api/attack/${encodeURIComponent(name)}/stop`, {});
    },

    stopAllAttacks() {
      return this.post("/api/attack/stop_all", {});
    },

    spawnObstacle(distance = 90.0, lifetime = 10.0) {
      return this.post("/api/scenario/real_obstacle", { distance, lifetime });
    },

    setProtection(enabled) {
      return this.post("/api/protection", { enabled });
    },

    reset() {
      return this.post("/api/reset", {});
    },

    startDemo() {
      return this.post("/api/demo/start", {});
    },

    stopDemo() {
      return this.post("/api/demo/stop", {});
    },

    async fetchSnapshot() {
      try {
        const res = await fetch("/api/snapshot");
        if (res.ok) return await res.json();
      } catch (err) {
        console.debug("Polling snapshot fetch failed:", err);
      }
      return null;
    },
  };

  // -------------------------------------------------------------------------
  // Initialize Dynamic Controls (Left Column Attack Buttons)
  // -------------------------------------------------------------------------
  function initControls() {
    dom.attackBtnGroup.innerHTML = "";

    ATTACKS.forEach((atk) => {
      const btn = document.createElement("button");
      btn.className = "btn-soc btn-attack";
      btn.id = `btn_atk_${atk.name}`;
      btn.dataset.attackName = atk.name;

      const labelSpan = createTextEl("span", atk.label);
      const iconSpan = createTextEl("span", "▶", "atk-icon");
      iconSpan.style.fontSize = "10px";
      iconSpan.style.opacity = "0.7";

      btn.appendChild(labelSpan);
      btn.appendChild(iconSpan);

      btn.addEventListener("click", () => {
        if (state.activeAttacks.has(atk.name)) {
          api.stopAttack(atk.name);
        } else {
          api.startAttack(atk.name);
        }
      });

      dom.attackBtnGroup.appendChild(btn);
    });

    // Real Obstacle button
    dom.btnRealObstacle.addEventListener("click", () => {
      api.spawnObstacle(90.0, 10.0);
    });

    // Protection switch button
    dom.btnProtectionToggle.addEventListener("click", () => {
      api.setProtection(!state.protection);
    });

    // Reset button
    dom.btnResetSystem.addEventListener("click", () => {
      api.reset();
    });

    // Guided Demo toggle button
    dom.btnGuidedDemo.addEventListener("click", () => {
      if (state.demoRunning) {
        api.stopDemo();
      } else {
        api.startDemo();
      }
    });

    // CAN Stream filter chips
    if (dom.canFilterAll && dom.canFilterFlagged) {
      dom.canFilterAll.addEventListener("click", () => {
        state.canFilter = "all";
        dom.canFilterAll.classList.add("active");
        dom.canFilterFlagged.classList.remove("active");
        state.lastJson.can = null;
        if (state.lastSnapshot) renderCANStream(state.lastSnapshot);
      });
      dom.canFilterFlagged.addEventListener("click", () => {
        state.canFilter = "flagged";
        dom.canFilterFlagged.classList.add("active");
        dom.canFilterAll.classList.remove("active");
        state.lastJson.can = null;
        if (state.lastSnapshot) renderCANStream(state.lastSnapshot);
      });
    }

    // V2X Stream filter chips
    if (dom.v2xFilterAll && dom.v2xFilterFlagged) {
      dom.v2xFilterAll.addEventListener("click", () => {
        state.v2xFilter = "all";
        dom.v2xFilterAll.classList.add("active");
        dom.v2xFilterFlagged.classList.remove("active");
        state.lastJson.v2x = null;
        if (state.lastSnapshot) renderV2XStream(state.lastSnapshot);
      });
      dom.v2xFilterFlagged.addEventListener("click", () => {
        state.v2xFilter = "flagged";
        dom.v2xFilterFlagged.classList.add("active");
        dom.v2xFilterAll.classList.remove("active");
        state.lastJson.v2x = null;
        if (state.lastSnapshot) renderV2XStream(state.lastSnapshot);
      });
    }
  }

  // -------------------------------------------------------------------------
  // Section Renderers (Optimized: Updates DOM only when JSON snapshot changes)
  // -------------------------------------------------------------------------

  // 1. Header, Mode Banner & Demo Caption
  function renderBanner(snap) {
    const rawBanner = JSON.stringify({
      mode: snap.mode,
      prot: snap.protection,
      banner: snap.banner,
      backend: snap.backend,
      demo: snap.demo,
    });
    if (rawBanner === state.lastJson.banner) return;
    state.lastJson.banner = rawBanner;

    // Header Chips
    if (snap.backend) {
      if (snap.backend.bus) dom.chipBus.textContent = `Bus: ${snap.backend.bus}`;
      if (snap.backend.ai) dom.chipAi.textContent = `AI: ${snap.backend.ai}`;
    }

    // Mode Banner Styling
    const mode = snap.mode || "NORMAL";
    const modeKey = mode.toLowerCase().replace(/\s+/g, "");
    dom.modeBanner.className = `mode-banner mode-${modeKey}`;
    dom.modePill.textContent = mode;

    if (snap.banner) {
      dom.bannerHeadline.textContent = snap.banner.headline || "System operational.";
      dom.bannerAction.textContent = snap.banner.driver_action ? `Action: ${snap.banner.driver_action}` : "";
    }

    // Protection Badge & Switch
    state.protection = Boolean(snap.protection);
    if (state.protection) {
      dom.protBadge.className = "prot-badge on";
      dom.protBadge.textContent = "Protection: ON (Enforcing)";
      dom.btnProtectionToggle.className = "btn-soc btn-switch on";
      dom.btnProtectionToggle.setAttribute("aria-checked", "true");
      dom.btnProtectionStatusText.textContent = "ON";
    } else {
      dom.protBadge.className = "prot-badge off";
      dom.protBadge.textContent = "Protection: OFF (Observe Only)";
      dom.btnProtectionToggle.className = "btn-soc btn-switch off";
      dom.btnProtectionToggle.setAttribute("aria-checked", "false");
      dom.btnProtectionStatusText.textContent = "OFF";
    }

    // Guided Demo Status
    const demo = snap.demo || {};
    state.demoRunning = Boolean(demo.running);
    if (state.demoRunning) {
      dom.demoCaptionBar.className = "demo-caption-bar active";
      dom.demoCaptionText.textContent = demo.caption || "Demo active...";
      dom.btnGuidedDemo.textContent = "■ Stop Demo";
      dom.btnGuidedDemo.style.background = "#b91c1c";
    } else {
      dom.demoCaptionBar.className = "demo-caption-bar";
      dom.btnGuidedDemo.textContent = "► Run Guided Demo (60s)";
      dom.btnGuidedDemo.style.background = "";
    }
  }

  // 2. Adversary Control Buttons Active Status
  function renderControls(snap) {
    const activeList = snap.active_attacks || [];
    state.activeAttacks = new Set(activeList);
    dom.attackCountBadge.textContent = `${activeList.length} Active`;

    ATTACKS.forEach((atk) => {
      const btn = document.getElementById(`btn_atk_${atk.name}`);
      if (!btn) return;
      const isActive = state.activeAttacks.has(atk.name);

      if (isActive) {
        if (!btn.classList.contains("active")) {
          btn.classList.add("active");
          btn.children[0].textContent = `Stop: ${atk.label}`;
          btn.children[1].textContent = "■";
        }
      } else {
        if (btn.classList.contains("active")) {
          btn.classList.remove("active");
          btn.children[0].textContent = atk.label;
          btn.children[1].textContent = "▶";
        }
      }
    });
  }

  // 3. Vehicle Digital Twin Telemetry Cards
  function renderDigitalTwin(snap) {
    const twin = snap.twin || {};
    const rawTwin = JSON.stringify({
      twin: twin,
      starvation: snap.starvation_warning,
    });
    if (rawTwin === state.lastJson.twin) return;
    state.lastJson.twin = rawTwin;

    // Speed Card & Perceived Discrepancy
    const speed = twin.speed_kmh != null ? twin.speed_kmh.toFixed(1) : "0.0";
    dom.twinSpeed.textContent = `${speed} km/h`;

    const perceived = twin.perceived || {};
    const pSpeed = perceived.speed_kmh;

    if (pSpeed != null) {
      const pSpeedStr = pSpeed.toFixed(1);
      const speedDelta = Math.abs((twin.speed_kmh || 0) - pSpeed);
      if (speedDelta > 5.0) {
        dom.twinSpeedSub.innerHTML = `<span class="spoofed-speed-alert">SPOOFED SPEED SHOWN: ${pSpeedStr} km/h</span> (True: ${speed})`;
      } else {
        dom.twinSpeedSub.textContent = `Cluster: ${pSpeedStr} km/h (True: ${speed} km/h)`;
      }
    } else {
      dom.twinSpeedSub.textContent = `Cluster: -- (stale) | True: ${speed} km/h`;
    }

    // RPM
    const rpm = twin.rpm != null ? Math.round(twin.rpm) : "0";
    dom.twinRpm.textContent = `${rpm} RPM`;
    const pedal = twin.pedal != null ? twin.pedal.toFixed(0) : "0";
    dom.twinRpmSub.textContent = `Throttle: ${pedal}%`;

    // Decel / Brake
    const decel = twin.applied_decel != null ? twin.applied_decel.toFixed(1) : "0.0";
    dom.twinDecel.textContent = `${decel} m/s²`;
    if (snap.starvation_warning) {
      dom.twinBrakeSub.innerHTML = `<span class="spoofed-speed-alert">FAIL-SAFE STARVATION (>200ms)</span>`;
    } else {
      const aebText = twin.aeb ? "AEB ACTIVE" : "Standby";
      dom.twinBrakeSub.textContent = `Pedal: ${pedal}% | AEB: ${aebText}`;
    }

    // Steering
    const steer = twin.steering_deg != null ? twin.steering_deg.toFixed(1) : "0.0";
    dom.twinSteering.textContent = `${steer}°`;

    // Gear
    const gearMap = { 0: "P", 1: "R", 2: "N", 3: "D" };
    const gearStr = gearMap[twin.gear] || (twin.gear != null ? String(twin.gear) : "D");
    dom.twinGear.textContent = gearStr;
    dom.twinGearSub.textContent = gearStr === "R" ? "Reverse Gear" : "Drive (Gear 3)";

    // Vehicle Safety Condition
    const isUnsafe = twin.unsafe && twin.unsafe.trim() !== "";
    const unsafeSec = twin.unsafe_seconds != null ? twin.unsafe_seconds.toFixed(1) : "0.0";

    if (isUnsafe) {
      dom.twinSafetyCard.className = "twin-card safety-card unsafe";
      dom.twinSafetyVal.textContent = "UNSAFE";
      dom.twinSafetySub.textContent = `${twin.unsafe} (${unsafeSec}s)`;
    } else {
      dom.twinSafetyCard.className = "twin-card safety-card safe";
      dom.twinSafetyVal.textContent = "SAFE";
      dom.twinSafetySub.textContent = `${unsafeSec}s cumulative unsafe`;
    }
  }

  // 4. Threat Monitoring Chart (Plotly Offline) & KPIs
  function renderThreatMonitoring(snap) {
    const threat = snap.threat || {};
    const rawKpis = JSON.stringify({
      threat: threat,
      markers: snap.attack_markers,
    });

    if (rawKpis !== state.lastJson.threatKpis) {
      state.lastJson.threatKpis = rawKpis;

      // KPIs
      dom.kpiAttackType.textContent = threat.attack_type || "Normal";
      const confPct = threat.confidence != null ? (threat.confidence * 100).toFixed(1) : "0.0";
      dom.kpiConfidence.textContent = `${confPct}%`;

      const level = threat.level || "LOW";
      dom.kpiRiskLevel.textContent = level;
      dom.kpiRiskLevel.style.color =
        level === "CRITICAL" ? "var(--col-safemode)" :
        level === "HIGH" ? "var(--col-degraded)" :
        level === "MEDIUM" ? "var(--col-alert)" : "var(--col-normal)";

      // Explanation Box
      const levelClass = `explanation-box level-${level.toLowerCase()}`;
      dom.explanationBox.className = levelClass;

      if (level === "CRITICAL" || level === "HIGH") {
        dom.explanationTitle.textContent = "Why it was blocked:";
      } else if (level === "MEDIUM") {
        dom.explanationTitle.textContent = "Caution: Elevated protocol activity";
      } else {
        dom.explanationTitle.textContent = "All clear: no inconsistencies found";
      }
      dom.explanationBody.textContent = threat.explanation || "All signals within verified invariants.";
    }

    // Chart: Limit Plotly updates to at most 2 times/sec
    const now = performance.now();
    if (now - state.lastChartRenderTime < 500) return;
    state.lastChartRenderTime = now;

    if (typeof Plotly === "undefined") return;

    const rawScores = snap.scores || [];
    const tVals = [];
    const anomVals = [];
    const atkProbVals = [];

    // Downsample to at most 200 points
    if (rawScores.length > 200) {
      const step = rawScores.length / 200;
      for (let i = 0; i < 200; i++) {
        const pt = rawScores[Math.floor(i * step)];
        tVals.push(pt.t);
        anomVals.push(pt.anomaly);
        atkProbVals.push(pt.attack_prob);
      }
    } else {
      rawScores.forEach((pt) => {
        tVals.push(pt.t);
        anomVals.push(pt.anomaly);
        atkProbVals.push(pt.attack_prob);
      });
    }

    const currentT = tVals.length > 0 ? tVals[tVals.length - 1] : 0;
    const xMin = Math.max(0, currentT - 30);
    const xMax = Math.max(30, currentT);

    const traces = [
      {
        name: "Anomaly",
        x: tVals,
        y: anomVals,
        mode: "lines",
        line: { color: "#38bdf8", width: 2 },
      },
      {
        name: "Attack Prob",
        x: tVals,
        y: atkProbVals,
        mode: "lines",
        line: { color: "#f59e0b", width: 1.5, dash: "dot" },
      },
    ];

    // Threshold line at 0.50
    const shapes = [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: 0.5,
        y1: 0.5,
        line: { color: "#ef4444", width: 1.5, dash: "dash" },
      },
    ];

    // Add vertical lines for attack markers
    const markers = snap.attack_markers || [];
    markers.forEach((mTime) => {
      shapes.push({
        type: "line",
        xref: "x",
        x0: mTime,
        x1: mTime,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color: "rgba(239, 68, 68, 0.4)", width: 1 },
      });
    });

    const layout = {
      height: 220,
      margin: { t: 8, b: 24, l: 32, r: 8 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "rgba(15, 22, 35, 0.6)",
      showlegend: false,
      uirevision: "constant",
      xaxis: {
        range: [xMin, xMax],
        color: "#64748b",
        tickfont: { size: 9, family: "monospace" },
        gridcolor: "#1e293b",
        zerolinecolor: "#1e293b",
      },
      yaxis: {
        range: [0, 1.0],
        color: "#64748b",
        tickfont: { size: 9, family: "monospace" },
        gridcolor: "#1e293b",
        zerolinecolor: "#1e293b",
        dtick: 0.25,
      },
      shapes: shapes,
    };

    const config = { displayModeBar: false, responsive: false };

    if (!state.plotlyInitialized) {
      Plotly.newPlot("threatPlotly", traces, layout, config);
      state.plotlyInitialized = true;
    } else {
      Plotly.react("threatPlotly", traces, layout, config);
    }
  }

  // 5. FailSafe Gateway Status & Incidents Table
  function renderFailSafe(snap) {
    const fs = snap.failsafe || {};
    const rawFs = JSON.stringify(fs);
    if (rawFs !== state.lastJson.failsafe) {
      state.lastJson.failsafe = rawFs;

      // Incident Counter
      dom.failsafeIncidentCount.textContent = `Incidents: ${fs.incident_id || 0}`;

      // Blocked Commands
      const rejCmds = fs.rejected_commands || {};
      const rejKeys = Object.keys(rejCmds);
      if (rejKeys.length > 0) {
        dom.statRejectedCommands.textContent = rejKeys.map((k) => `${k}: ${rejCmds[k]}`).join(", ");
        dom.statRejectedCommands.style.color = "var(--col-safemode)";
      } else {
        dom.statRejectedCommands.textContent = "None";
        dom.statRejectedCommands.style.color = "var(--text-main)";
      }

      // Rate Limited IDs
      const rlList = fs.rate_limited || [];
      dom.statRateLimited.textContent = rlList.length > 0 ? rlList.join(", ") : "None";

      // Quarantined IDs
      const qList = fs.quarantined || [];
      dom.statQuarantined.textContent = qList.length > 0 ? qList.join(", ") : "None";

      // Rejected V2X
      dom.statRejectedV2x.textContent = String(fs.rejected_v2x || 0);

      // Actions list
      const actions = fs.actions || [];
      dom.actionsList.innerHTML = "";
      if (actions.length === 0) {
        dom.actionsList.appendChild(createTextEl("div", "No gateway actions recorded.", "action-item"));
      } else {
        actions.slice(-8).reverse().forEach((act) => {
          dom.actionsList.appendChild(createTextEl("div", act, "action-item"));
        });
      }
    }

    // Incidents Table
    const incidents = snap.incidents || [];
    const rawIncidents = JSON.stringify(incidents);
    if (rawIncidents !== state.lastJson.incidents) {
      state.lastJson.incidents = rawIncidents;
      dom.incidentsTbody.innerHTML = "";

      incidents.slice(0, 10).forEach((inc) => {
        const tr = document.createElement("tr");

        tr.appendChild(createTextEl("td", `#${inc.id}`));
        tr.appendChild(createTextEl("td", inc.ts != null ? `${inc.ts.toFixed(1)}s` : "-"));

        // Level pill
        const lvlTd = document.createElement("td");
        const lvl = inc.level || "LOW";
        const lvlPill = createTextEl("span", lvl, `pill pill-level-${lvl.toLowerCase()}`);
        lvlTd.appendChild(lvlPill);
        tr.appendChild(lvlTd);

        tr.appendChild(createTextEl("td", inc.attack_type || "Unknown"));
        tr.appendChild(createTextEl("td", inc.target || "-"));
        tr.appendChild(createTextEl("td", inc.confidence != null ? `${(inc.confidence * 100).toFixed(0)}%` : "-"));
        tr.appendChild(createTextEl("td", inc.latency_ms != null ? `${inc.latency_ms.toFixed(1)}ms` : "-"));
        tr.appendChild(createTextEl("td", String(inc.count || 1)));

        dom.incidentsTbody.appendChild(tr);
      });
    }
  }

  // 6. Live CAN Telemetry Stream
  function renderCANStream(snap) {
    const rawList = snap.can || [];
    const flaggedCount = snap.can_flagged_count != null ? snap.can_flagged_count : rawList.filter(f => f.verdict !== "normal").length;
    if (dom.canFlaggedBadge) dom.canFlaggedBadge.textContent = String(flaggedCount);

    const displayList = state.canFilter === "flagged"
      ? rawList.filter((f) => (f.verdict || "").toLowerCase() !== "normal")
      : rawList;

    const cacheKey = JSON.stringify({ f: state.canFilter, list: displayList });
    if (cacheKey === state.lastJson.can) return;
    state.lastJson.can = cacheKey;

    dom.canTbody.innerHTML = "";

    displayList.slice(0, 15).forEach((f) => {
      const tr = document.createElement("tr");
      const v = (f.verdict || "normal").toLowerCase();

      if (v === "blocked") tr.className = "row-blocked";
      else if (v === "suspicious" || v === "forwarded") tr.className = "row-suspicious";

      tr.appendChild(createTextEl("td", f.ts != null ? f.ts.toFixed(3) : "-"));

      // ID with repeat-count formatting
      const idTd = document.createElement("td");
      idTd.appendChild(document.createTextNode(f.id || "-"));
      if (f.count && f.count > 1) {
        const rep = createTextEl("span", `(x ${f.count.toLocaleString()})`, "repeat-badge");
        idTd.appendChild(rep);
      }
      tr.appendChild(idTd);

      tr.appendChild(createTextEl("td", String(f.dlc || 8)));
      tr.appendChild(createTextEl("td", f.data || "-"));
      tr.appendChild(createTextEl("td", f.sender || "-"));

      // Verdict Pill
      const vTd = document.createElement("td");
      const vPill = createTextEl("span", f.verdict || "normal", `pill pill-${v}`);
      vTd.appendChild(vPill);
      tr.appendChild(vTd);

      dom.canTbody.appendChild(tr);
    });
  }

  // 7. Live V2X Telemetry Stream
  function renderV2XStream(snap) {
    const rawList = snap.v2x || [];
    const flaggedCount = snap.v2x_flagged_count != null ? snap.v2x_flagged_count : rawList.filter(m => m.status !== "accepted").length;
    if (dom.v2xFlaggedBadge) dom.v2xFlaggedBadge.textContent = String(flaggedCount);

    const displayList = state.v2xFilter === "flagged"
      ? rawList.filter((m) => (m.status || "").toLowerCase() !== "accepted")
      : rawList;

    const cacheKey = JSON.stringify({ f: state.v2xFilter, list: displayList });
    if (cacheKey === state.lastJson.v2x) return;
    state.lastJson.v2x = cacheKey;

    dom.v2xTbody.innerHTML = "";

    displayList.slice(0, 12).forEach((msg) => {
      const tr = document.createElement("tr");
      const st = (msg.status || "accepted").toLowerCase();

      if (st === "rejected") tr.className = "row-blocked";
      else if (st === "suspicious" || st === "forwarded") tr.className = "row-suspicious";

      tr.appendChild(createTextEl("td", msg.ts != null ? msg.ts.toFixed(3) : "-"));

      // Sender with repeat-count formatting
      const senderTd = document.createElement("td");
      senderTd.appendChild(document.createTextNode(msg.sender || "-"));
      if (msg.count && msg.count > 1) {
        const rep = createTextEl("span", `(x ${msg.count.toLocaleString()})`, "repeat-badge");
        senderTd.appendChild(rep);
      }
      tr.appendChild(senderTd);

      tr.appendChild(createTextEl("td", msg.type || "-"));

      const trust = msg.trust != null ? (msg.trust * 100).toFixed(0) + "%" : "-";
      tr.appendChild(createTextEl("td", trust));

      const stTd = document.createElement("td");
      const stPill = createTextEl("span", msg.status || "accepted", `pill pill-${st}`);
      stTd.appendChild(stPill);

      if ((st === "rejected" || st === "suspicious" || st === "forwarded") && msg.reason) {
        const reasonSub = createTextEl("span", msg.reason, "v2x-reason-sub");
        stTd.appendChild(reasonSub);
      }
      tr.appendChild(stTd);

      dom.v2xTbody.appendChild(tr);
    });
  }

  // 7b. Security Events Feed (Combined CAN + V2X)
  function renderSecurityEvents(snap) {
    if (!dom.securityEventsTbody) return;
    const events = snap.security_events || [];
    const rawEvents = JSON.stringify(events);
    if (rawEvents === state.lastJson.securityEvents) return;
    state.lastJson.securityEvents = rawEvents;

    if (dom.secEventsCount) {
      dom.secEventsCount.textContent = `${events.length} Events`;
    }

    dom.securityEventsTbody.innerHTML = "";

    if (events.length === 0) {
      const tr = document.createElement("tr");
      const td = createTextEl("td", "No security incidents detected. System secure.");
      td.colSpan = 5;
      td.style.color = "var(--text-dim)";
      td.style.textAlign = "center";
      tr.appendChild(td);
      dom.securityEventsTbody.appendChild(tr);
      return;
    }

    events.slice(0, 20).forEach((ev) => {
      const tr = document.createElement("tr");
      const isCan = ev.domain === "CAN";
      const v = (ev.verdict || "blocked").toLowerCase();

      if (v === "blocked" || v === "rejected") tr.className = "row-blocked";
      else if (v === "suspicious") tr.className = "row-suspicious";

      // Domain Badge
      const dTd = document.createElement("td");
      const dBadge = createTextEl("span", `[${ev.domain}]`, isCan ? "domain-badge-can" : "domain-badge-v2x");
      dTd.appendChild(dBadge);
      tr.appendChild(dTd);

      // Time
      tr.appendChild(createTextEl("td", ev.ts != null ? `${ev.ts.toFixed(3)}s` : "-"));

      // ID / Sender
      const idTd = document.createElement("td");
      idTd.appendChild(document.createTextNode(ev.id || ev.sender || "-"));
      if (ev.count && ev.count > 1) {
        const rep = createTextEl("span", `(x ${ev.count.toLocaleString()})`, "repeat-badge");
        idTd.appendChild(rep);
      }
      tr.appendChild(idTd);

      // Verdict / Status Pill
      const vTd = document.createElement("td");
      const vPill = createTextEl("span", ev.verdict || "blocked", `pill pill-${v}`);
      vTd.appendChild(vPill);
      tr.appendChild(vTd);

      // Reason / Explanation
      tr.appendChild(createTextEl("td", ev.reason || "-"));

      dom.securityEventsTbody.appendChild(tr);
    });
  }

  // 8. Evaluation Metrics Strip (8 Cards)
  function renderMetrics(snap) {
    const m = snap.metrics || {};
    const rawMetrics = JSON.stringify(m);
    if (rawMetrics === state.lastJson.metrics) return;
    state.lastJson.metrics = rawMetrics;

    const fmtPct = (val) => (val != null ? `${(val * 100).toFixed(1)}%` : "-");
    const fmtNum = (val) => (val != null ? val.toFixed(4) : "-");

    dom.metricAcc.textContent = fmtPct(m.accuracy);
    dom.metricPrec.textContent = fmtPct(m.precision);
    dom.metricRec.textContent = fmtPct(m.recall);
    dom.metricF1.textContent = fmtNum(m.f1);
    dom.metricFpr.textContent = fmtPct(m.fpr);

    // Latencies: mean / p95
    if (m.detection_latency_ms) {
      dom.metricDetLat.textContent = `${m.detection_latency_ms.mean.toFixed(2)} ms`;
      dom.metricDetLatSub.textContent = `p95: ${m.detection_latency_ms.p95.toFixed(2)} ms | Max: ${m.detection_latency_ms.max.toFixed(1)} ms`;
    }

    if (m.failsafe_latency_ms) {
      dom.metricFsLat.textContent = `${m.failsafe_latency_ms.mean.toFixed(2)} ms`;
      dom.metricFsLatSub.textContent = `p95: ${m.failsafe_latency_ms.p95.toFixed(2)} ms`;
    }

    if (m.inference_ms) {
      dom.metricInfLat.textContent = `${m.inference_ms.mean.toFixed(1)} ms`;
      dom.metricInfLatSub.textContent = `p95: ${m.inference_ms.p95.toFixed(1)} ms`;
    }
  }

  // -------------------------------------------------------------------------
  // Master Snapshot Dispatcher (Batched in requestAnimationFrame)
  // -------------------------------------------------------------------------
  function applySnapshot(snap) {
    if (!snap) return;
    state.lastSnapshot = snap;
    requestAnimationFrame(() => {
      renderBanner(snap);
      renderControls(snap);
      renderDigitalTwin(snap);
      renderThreatMonitoring(snap);
      renderFailSafe(snap);
      renderCANStream(snap);
      renderV2XStream(snap);
      renderSecurityEvents(snap);
      renderMetrics(snap);
    });
  }

  // -------------------------------------------------------------------------
  // WebSocket Connection with Polling Fallback & Exponential Backoff
  // -------------------------------------------------------------------------
  let ws = null;
  let pollInterval = null;
  let reconnectTimeout = null;
  let reconnectDelay = 1000;

  function connectWebSocket() {
    clearTimeout(reconnectTimeout);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "127.0.0.1:8000";
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        dom.chipConn.className = "chip chip-conn live";
        dom.connStatusText.textContent = "Live (WS)";
        reconnectDelay = 1000;
        if (pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const snap = JSON.parse(event.data);
          applySnapshot(snap);
        } catch (err) {
          console.error("Malformed WebSocket JSON payload:", err);
        }
      };

      ws.onerror = (err) => {
        console.debug("WebSocket error:", err);
      };

      ws.onclose = () => {
        dom.chipConn.className = "chip chip-conn reconnecting";
        dom.connStatusText.textContent = "Reconnecting";

        // Fall back to HTTP polling if WS is down
        if (!pollInterval) {
          pollInterval = setInterval(async () => {
            const snap = await api.fetchSnapshot();
            if (snap) applySnapshot(snap);
          }, 500);
        }

        // Exponential backoff reconnect
        reconnectDelay = Math.min(reconnectDelay * 1.5, 8000);
        reconnectTimeout = setTimeout(connectWebSocket, reconnectDelay);
      };
    } catch (err) {
      console.warn("Could not open WebSocket; polling /api/snapshot:", err);
      if (!pollInterval) {
        pollInterval = setInterval(async () => {
          const snap = await api.fetchSnapshot();
          if (snap) applySnapshot(snap);
        }, 500);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Resize Observer for Plotly (Debounced to 250ms)
  // -------------------------------------------------------------------------
  function initChartResizeObserver() {
    const el = document.getElementById("threatPlotly");
    if (!el || typeof ResizeObserver === "undefined") return;
    let resizeTimer = null;
    const observer = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.plotlyInitialized && typeof Plotly !== "undefined") {
          Plotly.Plots.resize(el);
        }
      }, 250);
    });
    observer.observe(el);
  }

  // -------------------------------------------------------------------------
  // Application Entry Point
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initControls();
    initChartResizeObserver();
    connectWebSocket();
  });
})();

