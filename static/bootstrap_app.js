const languageSelectEl = document.getElementById("language-select");

const LANG_STORAGE_KEY = "dblpservice_lang";
const SUPPORTED_LANGS = new Set(["en", "zh"]);
let currentLang = "en";
const sidebarEl = document.getElementById("sidebar");
const sidebarToggleEl = document.getElementById("sidebar-toggle");
const sidebarOverlayEl = document.getElementById("sidebar-overlay");

const I18N = {
  en: {
    page_title_bootstrap: "DblpService Bootstrap Console",
    brand_tagline:
      "Build and maintain DBLP SQLite, and provide unified query APIs for CoAuthors and CiteVerifier",
    brand_chip: "Maintained by AOSP Laboratory",
    control_language: "Language",
    lang_en: "English",
    lang_zh: "Chinese",
    bootstrap_title: "DblpService Bootstrap Console",
    bootstrap_subtitle:
      "Download XML/DTD, decompress XML, and build DBLP SQLite through the web.",
    bootstrap_params_title: "Task Parameters",
    bootstrap_xml_url: "XML.GZ URL",
    bootstrap_dtd_url: "DTD URL",
    bootstrap_batch: "Batch Size",
    bootstrap_progress_every: "Progress Every",
    bootstrap_rebuild: "Rebuild database (remove existing sqlite/wal/shm)",
    bootstrap_start: "Start",
    bootstrap_stop: "Stop",
    bootstrap_reset: "Reset",
    bootstrap_runtime: "Runtime Status",
    bootstrap_progress: "Progress",
    bootstrap_output_files: "Output Files",
    bootstrap_logs: "Live Logs",
    runtime_status: "Status",
    runtime_step: "Step",
    runtime_message: "Message",
    runtime_started: "Started",
    runtime_finished: "Finished",
    progress_downloaded: "Downloaded",
    progress_total: "Total",
    progress_xml_written: "XML Written",
    progress_records: "Processed Records",
    progress_rate: "Rate",
    progress_data_dir: "Data Dir",
    table_file: "File",
    table_exists: "Exists",
    table_size: "Size",
    table_path: "Path",
    yes: "yes",
    no: "no",
    msg_refresh_failed: "Refresh failed: {err}",
    msg_start_failed: "Start failed: {err}",
    msg_stop_failed: "Stop failed: {err}",
    msg_reset_failed: "Reset failed: {err}",
    footer_title: "Project Information",
    footer_owner_label: "Developer & Maintainer",
    footer_owner_value: "Nankai University AOSP Laboratory",
    footer_dev_label: "Developer",
    footer_dev_value: "Nankai University AOSP Laboratory",
    footer_maintainer_label: "Maintainer",
    footer_maintainer_value: "Nankai University AOSP Laboratory",
    footer_members_label: "Members",
    footer_members_value: "Xiang Li, Zuyao Xu, Yuqi Qiu, Fubin Wu, Fasheng Miao, Lu Sun",
    footer_version_label: "Version",
    footer_features_label: "Current Features",
    footer_features_value:
      "DBLP download, decompression, database rebuild, and runtime status monitoring.",
    footer_license_label: "License",
    footer_visits_label: "Visits",
    footer_copyright_label: "Copyright",
    footer_copyright_value:
      "\u00A9 2026 AOSP Lab of Nankai University. All Rights Reserved.",
    lab_name: "AOSP Laboratory, Nankai University",
    lab_slogan: "All-in-One Security and Privacy Lab",
    lab_description:
      "The lab focuses on diversified security and privacy research, spanning network security, Web security, LLM security, and emerging security risks. It is dedicated to enhancing overall security in scenarios where network technologies converge with large models, and continuously supports the security community through original research contributions.",
    lab_advisor_intro:
      "Advisor: <a href=\"https://lixiang521.com/\" target=\"_blank\" rel=\"noopener\">Xiang Li</a>, Associate Professor at the College of Cryptology and Cyber Science, Nankai University. National Outstanding Talent in Cyberspace. Research areas include network security, protocol security, vulnerability discovery, and LLM security.",
    lab_qrcode_caption: "Follow AOSP Lab",
  },
  zh: {
    page_title_bootstrap: "DblpService 建库控制台",
    brand_tagline:
      "负责 DBLP SQLite 建库维护，并为 CoAuthors 与 CiteVerifier 提供统一查询 API",
    brand_chip: "由AOSP实验室维护",
    control_language: "语言",
    lang_en: "英文",
    lang_zh: "中文",
    bootstrap_title: "DblpService 建库控制台",
    bootstrap_subtitle: "通过 Web 页面完成 XML/DTD 下载、XML 解压与 DBLP SQLite 建库。",
    bootstrap_params_title: "任务参数",
    bootstrap_xml_url: "XML.GZ 地址",
    bootstrap_dtd_url: "DTD 地址",
    bootstrap_batch: "批处理大小",
    bootstrap_progress_every: "进度上报间隔",
    bootstrap_rebuild: "重建数据库（删除已有 sqlite/wal/shm）",
    bootstrap_start: "开始",
    bootstrap_stop: "停止",
    bootstrap_reset: "重置",
    bootstrap_runtime: "运行状态",
    bootstrap_progress: "进度",
    bootstrap_output_files: "输出文件",
    bootstrap_logs: "实时日志",
    runtime_status: "状态",
    runtime_step: "步骤",
    runtime_message: "消息",
    runtime_started: "开始时间",
    runtime_finished: "结束时间",
    progress_downloaded: "已下载",
    progress_total: "总大小",
    progress_xml_written: "XML 写入",
    progress_records: "处理记录数",
    progress_rate: "速率",
    progress_data_dir: "数据目录",
    table_file: "文件",
    table_exists: "是否存在",
    table_size: "大小",
    table_path: "路径",
    yes: "是",
    no: "否",
    msg_refresh_failed: "刷新失败：{err}",
    msg_start_failed: "启动失败：{err}",
    msg_stop_failed: "停止失败：{err}",
    msg_reset_failed: "重置失败：{err}",
    footer_title: "项目信息",
    footer_owner_label: "开发与维护",
    footer_owner_value: "南开大学 AOSP 实验室",
    footer_dev_label: "开发团队",
    footer_dev_value: "南开大学 AOSP 实验室",
    footer_maintainer_label: "维护团队",
    footer_maintainer_value: "南开大学 AOSP 实验室",
    footer_members_label: "成员",
    footer_members_value: "李想，许祖耀，仇渝淇，吴福彬，苗发生，孙蕗",
    footer_version_label: "版本",
    footer_features_label: "当前特性",
    footer_features_value: "DBLP 下载、解压、数据库重建与运行状态监控。",
    footer_license_label: "开源协议",
    footer_visits_label: "访问量",
    footer_copyright_label: "版权",
    footer_copyright_value:
      "\u00A9 2026 AOSP Lab of Nankai University. All Rights Reserved.",
    lab_name: "南开大学 AOSP 实验室",
    lab_slogan: "All-in-One Security and Privacy Lab",
    lab_description:
      "实验室：聚焦多元化安全与隐私研究，涵盖网络安全、Web 安全、大模型安全及新兴安全风险等方向，致力于提升网络技术与大模型融合场景下的整体安全性，并通过原创性研究成果持续服务与支撑安全社区发展。",
    lab_advisor_intro:
      "导师：<a href=\"https://lixiang521.com/\" target=\"_blank\" rel=\"noopener\">李想</a>，南开大学密码与网络空间安全学院副教授，国家网信领域优秀人才，研究领域包括网络安全、协议安全、漏洞挖掘与大模型安全。",
    lab_qrcode_caption: "关注 AOSP 实验室",
  },
};

function t(key, vars = {}) {
  const langPack = I18N[currentLang] || I18N.en;
  const template = langPack[key] ?? I18N.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? `{${name}}`));
}

function fmtBytes(bytes) {
  if (typeof bytes !== "number" || Number.isNaN(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0;
  let val = bytes;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx += 1;
  }
  return `${val.toFixed(idx === 0 ? 0 : 2)} ${units[idx]}`;
}

function toggleSidebar() {
  if (!sidebarEl) return;
  const isOpen = sidebarEl.classList.toggle("is-open");
  if (sidebarOverlayEl) {
    sidebarOverlayEl.classList.toggle("is-active", isOpen);
  }
}

function closeSidebar() {
  if (sidebarEl) sidebarEl.classList.remove("is-open");
  if (sidebarOverlayEl) sidebarOverlayEl.classList.remove("is-active");
}

if (sidebarToggleEl) {
  sidebarToggleEl.addEventListener("click", toggleSidebar);
}

if (sidebarOverlayEl) {
  sidebarOverlayEl.addEventListener("click", closeSidebar);
}

function applyLanguage(lang) {
  currentLang = SUPPORTED_LANGS.has(lang) ? lang : "en";
  document.documentElement.lang = currentLang === "zh" ? "zh-CN" : "en";
  if (languageSelectEl && languageSelectEl.value !== currentLang) {
    languageSelectEl.value = currentLang;
  }

  try {
    window.localStorage.setItem(LANG_STORAGE_KEY, currentLang);
  } catch (_) {}

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (!key) return;
    if (el.tagName === "TITLE") {
      document.title = t(key);
      return;
    }
    el.textContent = t(key);
  });

  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const key = el.getAttribute("data-i18n-html");
    if (!key) return;
    el.innerHTML = t(key);
  });
}

function initLanguage() {
  let initialLang = "en";
  try {
    const savedLang = window.localStorage.getItem(LANG_STORAGE_KEY);
    if (savedLang && SUPPORTED_LANGS.has(savedLang)) {
      initialLang = savedLang;
    }
  } catch (_) {}

  applyLanguage(initialLang);
  if (languageSelectEl) {
    languageSelectEl.addEventListener("change", (event) => {
      applyLanguage(event.target.value);
      refreshAll();
    });
  }
}

async function fetchJson(url, options = {}) {
  const apiBaseRaw = String(window.__API_BASE__ || "").trim();
  const apiBase = apiBaseRaw.endsWith("/") ? apiBaseRaw.slice(0, -1) : apiBaseRaw;
  const fullUrl = apiBase ? `${apiBase}${url}` : url;
  const resp = await fetch(fullUrl, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return data;
}

function fillText(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value ?? "-";
}

function setStatusChip(status) {
  const el = document.getElementById("status");
  if (!el) return;
  const normalized = String(status || "").toLowerCase();
  el.className = "chip";
  if (normalized === "completed") {
    el.classList.add("chip-ok");
  } else if (normalized === "error") {
    el.classList.add("chip-error");
  } else {
    el.classList.add("chip-warn");
  }
}

function updateState(state) {
  fillText("status", state.status);
  setStatusChip(state.status);
  fillText("step", state.step);
  fillText("message", state.message || "-");
  fillText("started-at", state.started_at || "-");
  fillText("finished-at", state.finished_at || "-");

  const p = state.progress || {};
  fillText("downloaded-bytes", fmtBytes(p.downloaded_bytes));
  fillText("total-bytes", fmtBytes(p.total_bytes));
  fillText("written-bytes", fmtBytes(p.written_bytes));
  fillText("processed-records", p.processed_records ?? "-");
  fillText("records-rate", p.records_per_sec !== undefined ? `${p.records_per_sec} rec/s` : "-");

  const logs = Array.isArray(state.logs) ? state.logs : [];
  const logBox = document.getElementById("logs");
  if (logBox) {
    logBox.textContent = logs.join("\n");
    logBox.scrollTop = logBox.scrollHeight;
  }
}

function updateFiles(data) {
  fillText("data-dir", data.data_dir || "-");
  const tbody = document.getElementById("file-table");
  if (!tbody) return;
  tbody.innerHTML = "";

  const files = data.files || {};
  for (const [name, info] of Object.entries(files)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${name}</td>
      <td>${info.exists ? t("yes") : t("no")}</td>
      <td>${fmtBytes(info.size ?? 0)}</td>
      <td><code>${info.path || "-"}</code></td>
    `;
    tbody.appendChild(tr);
  }
}

async function refreshAll() {
  try {
    const [state, files] = await Promise.all([fetchJson("/api/state"), fetchJson("/api/files")]);
    updateState(state);
    updateFiles(files);
  } catch (err) {
    fillText("message", t("msg_refresh_failed", { err: err.message }));
  }
}

const startFormEl = document.getElementById("start-form");
if (startFormEl) {
  startFormEl.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const payload = {
      xml_gz_url: document.getElementById("xml-gz-url")?.value?.trim(),
      dtd_url: document.getElementById("dtd-url")?.value?.trim(),
      rebuild: Boolean(document.getElementById("rebuild")?.checked),
      batch_size: Number(document.getElementById("batch-size")?.value || 1000),
      progress_every: Number(document.getElementById("progress-every")?.value || 10000),
    };

    try {
      await fetchJson("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await refreshAll();
    } catch (err) {
      fillText("message", t("msg_start_failed", { err: err.message }));
    }
  });
}

const stopBtnEl = document.getElementById("stop-btn");
if (stopBtnEl) {
  stopBtnEl.addEventListener("click", async () => {
    try {
      await fetchJson("/api/stop", { method: "POST" });
      await refreshAll();
    } catch (err) {
      fillText("message", t("msg_stop_failed", { err: err.message }));
    }
  });
}

const resetBtnEl = document.getElementById("reset-btn");
if (resetBtnEl) {
  resetBtnEl.addEventListener("click", async () => {
    try {
      await fetchJson("/api/reset", { method: "POST" });
      await refreshAll();
    } catch (err) {
      fillText("message", t("msg_reset_failed", { err: err.message }));
    }
  });
}

initLanguage();
refreshAll();
setInterval(refreshAll, 2000);

