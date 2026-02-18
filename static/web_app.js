const languageSelectEl = document.getElementById("language-select");
const sidebarEl = document.getElementById("sidebar");
const sidebarToggleEl = document.getElementById("sidebar-toggle");
const sidebarOverlayEl = document.getElementById("sidebar-overlay");

const healthStatusEl = document.getElementById("health-status");
const maxCandidatesEl = document.getElementById("max-candidates");
const titleInputEl = document.getElementById("title-input");
const searchFormEl = document.getElementById("search-form");
const searchMsgEl = document.getElementById("search-msg");
const resultBoxEl = document.getElementById("result-box");

const batchFormEl = document.getElementById("batch-form");
const batchInputEl = document.getElementById("batch-input");
const batchFileEl = document.getElementById("batch-file");
const batchMsgEl = document.getElementById("batch-msg");
const batchSummaryEl = document.getElementById("batch-summary");
const batchTbodyEl = document.getElementById("batch-tbody");
const batchDownloadBtnEl = document.getElementById("batch-download-btn");

const MAX_BATCH_SIZE = 200;
const LANG_STORAGE_KEY = "citeverifier_lang";
const SUPPORTED_LANGS = new Set(["en", "zh"]);
let currentLang = "en";
let latestBatchItems = [];
let latestBatchSummary = null;
let latestSingleResult = null;
let latestHealthState = "checking";

const I18N = {
  en: {
    page_title: "CiteVerifier DBLP Search",
    brand_tagline: "Search and batch-verify paper titles against local DBLP data",
    brand_chip: "Maintained by AOSP Laboratory",
    control_language: "Language",
    lang_en: "English",
    lang_zh: "Chinese",
    status_title: "System Status",
    status_service: "Service",
    status_checking: "Checking",
    status_ok: "OK",
    status_error: "ERROR",
    settings_title: "Lookup Settings",
    settings_max_candidates: "Max Candidates",
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
      "Single-title lookup, batch verification, CSV export, and runtime telemetry.",
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

    hero_title: "CiteVerifier DBLP Title Search",
    hero_subtitle: "Input a paper title and search from local DBLP SQLite only.",
    search_title: "Search",
    search_input_label: "Paper Title",
    search_input_placeholder: "Attention Is All You Need",
    search_button: "Search",
    batch_title: "Batch Search",
    batch_input_label: "Titles (one per line)",
    batch_input_placeholder:
      "Attention Is All You Need\nBERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    batch_file_label: "Upload TXT/CSV (first column used)",
    batch_run_button: "Run Batch",
    batch_download_button: "Download CSV",
    batch_no_result: "No batch result yet.",
    result_title: "Result",
    result_empty: "No result yet.",

    table_index: "#",
    table_query_title: "Query Title",
    table_status: "Status",
    table_matched_title: "Matched Title",
    table_similarity: "Similarity",
    table_year: "Year",
    table_venue: "Venue",

    result_query_title: "Query Title",
    result_status: "Status",
    result_no_match_found: "No match found",
    result_matched_title: "Matched Title",
    result_similarity: "Similarity",
    result_dblp_id: "DBLP ID",
    result_year: "Year",
    result_venue: "Venue",
    result_pub_type: "Publication Type",

    summary_run_id: "Run ID",
    summary_processed: "Processed",
    summary_found: "Found",
    summary_not_found: "Not Found",
    summary_duration: "Duration",

    status_matched: "Matched",
    status_not_matched: "Not Matched",

    msg_health_failed: "Health check failed: {err}",
    msg_enter_title: "Please enter a paper title.",
    msg_searching: "Searching...",
    msg_match_found: "Match found.",
    msg_no_match: "No match found.",
    msg_search_failed: "Search failed: {err}",
    msg_file_read_failed: "Failed to read file: {err}",
    msg_batch_input_required: "Please input at least one title (textarea or file).",
    msg_batch_too_many: "Too many titles. Limit is {max}.",
    msg_batch_processing: "Processing {n} titles...",
    msg_batch_completed: "Batch completed. Matched {found} / {total}.",
    msg_batch_failed: "Batch failed: {err}",
    msg_no_rows: "No rows.",
  },
  zh: {
    page_title: "CiteVerifier DBLP 检索",
    brand_tagline: "基于本地 DBLP 数据进行论文标题检索与批量核验",
    brand_chip: "由AOSP实验室维护",
    control_language: "语言",
    lang_en: "英文",
    lang_zh: "中文",
    status_title: "系统状态",
    status_service: "服务",
    status_checking: "检测中",
    status_ok: "正常",
    status_error: "异常",
    settings_title: "检索设置",
    settings_max_candidates: "候选上限",
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
    footer_features_value: "单标题检索、批处理核验、CSV 导出与运行时遥测。",
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

    hero_title: "CiteVerifier DBLP 标题检索",
    hero_subtitle: "输入论文标题，仅基于本地 DBLP SQLite 完成检索。",
    search_title: "单条检索",
    search_input_label: "论文标题",
    search_input_placeholder: "Attention Is All You Need",
    search_button: "开始检索",
    batch_title: "批量检索",
    batch_input_label: "标题列表（每行一个）",
    batch_input_placeholder:
      "Attention Is All You Need\nBERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    batch_file_label: "上传 TXT/CSV（使用第一列）",
    batch_run_button: "执行批处理",
    batch_download_button: "下载 CSV",
    batch_no_result: "暂无批处理结果。",
    result_title: "检索结果",
    result_empty: "暂无结果。",

    table_index: "#",
    table_query_title: "查询标题",
    table_status: "状态",
    table_matched_title: "匹配标题",
    table_similarity: "相似度",
    table_year: "年份",
    table_venue: "会议/期刊",

    result_query_title: "查询标题",
    result_status: "状态",
    result_no_match_found: "未找到匹配",
    result_matched_title: "匹配标题",
    result_similarity: "相似度",
    result_dblp_id: "DBLP ID",
    result_year: "年份",
    result_venue: "会议/期刊",
    result_pub_type: "文献类型",

    summary_run_id: "运行 ID",
    summary_processed: "处理数",
    summary_found: "匹配数",
    summary_not_found: "未匹配数",
    summary_duration: "耗时",

    status_matched: "已匹配",
    status_not_matched: "未匹配",

    msg_health_failed: "健康检查失败：{err}",
    msg_enter_title: "请输入论文标题。",
    msg_searching: "检索中...",
    msg_match_found: "已找到匹配。",
    msg_no_match: "未找到匹配。",
    msg_search_failed: "检索失败：{err}",
    msg_file_read_failed: "读取文件失败：{err}",
    msg_batch_input_required: "请至少输入一个标题（文本框或文件）。",
    msg_batch_too_many: "标题数量过多，最多 {max} 条。",
    msg_batch_processing: "正在处理 {n} 条标题...",
    msg_batch_completed: "批处理完成：匹配 {found} / {total}。",
    msg_batch_failed: "批处理失败：{err}",
    msg_no_rows: "暂无数据行。",
  },
};

function t(key, vars = {}) {
  const langPack = I18N[currentLang] || I18N.en;
  const template = langPack[key] ?? I18N.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? `{${name}}`));
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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

function setMessage(text, isError = false) {
  searchMsgEl.textContent = text || "";
  searchMsgEl.classList.toggle("error", Boolean(isError));
}

function setBatchMessage(text, isError = false) {
  batchMsgEl.textContent = text || "";
  batchMsgEl.classList.toggle("error", Boolean(isError));
}

function setHealthChip(state) {
  latestHealthState = state;
  healthStatusEl.classList.remove("chip-ok", "chip-warn", "chip-error");
  if (state === "ok") {
    healthStatusEl.classList.add("chip-ok");
    healthStatusEl.textContent = t("status_ok");
  } else if (state === "error") {
    healthStatusEl.classList.add("chip-error");
    healthStatusEl.textContent = t("status_error");
  } else {
    healthStatusEl.classList.add("chip-warn");
    healthStatusEl.textContent = t("status_checking");
  }
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

  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (!key) return;
    el.setAttribute("placeholder", t(key));
  });

  setHealthChip(latestHealthState);

  if (latestSingleResult) {
    if (latestSingleResult.found) renderMatch(latestSingleResult);
    else renderNoMatch(latestSingleResult);
  } else {
    resultBoxEl.classList.add("empty");
    resultBoxEl.textContent = t("result_empty");
  }

  renderBatchSummary(latestBatchSummary);
  renderBatchTable(latestBatchItems);
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
    languageSelectEl.addEventListener("change", (event) => applyLanguage(event.target.value));
  }
}

function normalizeTitle(text) {
  return String(text || "")
    .trim()
    .replace(/\s+/g, " ");
}

function parseLines(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => normalizeTitle(line))
    .filter((line) => line.length > 0);
}

function parseFirstCsvColumn(line) {
  const raw = String(line || "");
  if (!raw.trim()) return "";

  if (!raw.startsWith('"')) {
    return normalizeTitle(raw.split(",")[0] || "");
  }

  let value = "";
  for (let i = 1; i < raw.length; i += 1) {
    const ch = raw[i];
    if (ch === '"') {
      const next = raw[i + 1];
      if (next === '"') {
        value += '"';
        i += 1;
        continue;
      }
      break;
    }
    value += ch;
  }
  return normalizeTitle(value);
}

function dedupeTitles(items) {
  const output = [];
  const seen = new Set();
  for (const item of items) {
    const title = normalizeTitle(item);
    if (!title) continue;
    const key = title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(title);
  }
  return output;
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

async function refreshHealth() {
  setHealthChip("checking");
  try {
    const data = await fetchJson("/api/health");
    if (data.status === "ok") setHealthChip("ok");
    else setHealthChip("error");
  } catch (err) {
    setHealthChip("error");
    setMessage(t("msg_health_failed", { err: err.message }), true);
  }
}

function renderNoMatch(data) {
  resultBoxEl.classList.remove("empty");
  resultBoxEl.innerHTML = `
    <div><span class="result-k">${t("result_query_title")}</span><span class="result-v">${escapeHtml(data.query_title || "-")}</span></div>
    <div><span class="result-k">${t("result_status")}</span><span class="result-v">${t("result_no_match_found")}</span></div>
  `;
}

function renderMatch(data) {
  const sim = Number(data.dblp_title_similarity);
  const simText = Number.isFinite(sim) ? sim.toFixed(4) : "-";
  resultBoxEl.classList.remove("empty");
  resultBoxEl.innerHTML = `
    <div><span class="result-k">${t("result_query_title")}</span><span class="result-v">${escapeHtml(data.query_title || "-")}</span></div>
    <div><span class="result-k">${t("result_matched_title")}</span><span class="result-v">${escapeHtml(data.dblp_title || "-")}</span></div>
    <div><span class="result-k">${t("result_similarity")}</span><span class="result-v">${simText}</span></div>
    <div><span class="result-k">${t("result_dblp_id")}</span><span class="result-v">${escapeHtml(data.dblp_id ?? "-")}</span></div>
    <div><span class="result-k">${t("result_year")}</span><span class="result-v">${escapeHtml(data.year ?? "-")}</span></div>
    <div><span class="result-k">${t("result_venue")}</span><span class="result-v">${escapeHtml(data.venue || "-")}</span></div>
    <div><span class="result-k">${t("result_pub_type")}</span><span class="result-v">${escapeHtml(data.pub_type || "-")}</span></div>
  `;
}

function renderBatchSummary(summary) {
  if (!summary) {
    batchSummaryEl.classList.add("empty");
    batchSummaryEl.textContent = t("batch_no_result");
    return;
  }
  batchSummaryEl.classList.remove("empty");
  batchSummaryEl.innerHTML = `
    <div><span class="result-k">${t("summary_run_id")}</span><span class="result-v">${escapeHtml(summary.run_id)}</span></div>
    <div><span class="result-k">${t("summary_processed")}</span><span class="result-v">${escapeHtml(summary.total_processed)} / ${escapeHtml(summary.total_input)}</span></div>
    <div><span class="result-k">${t("summary_found")}</span><span class="result-v">${escapeHtml(summary.found_count)}</span></div>
    <div><span class="result-k">${t("summary_not_found")}</span><span class="result-v">${escapeHtml(summary.not_found_count)}</span></div>
    <div><span class="result-k">${t("summary_duration")}</span><span class="result-v">${escapeHtml(summary.duration_ms)} ms</span></div>
  `;
}

function renderBatchTable(items) {
  batchTbodyEl.innerHTML = "";
  if (!items || items.length === 0) {
    const row = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = t("msg_no_rows");
    row.appendChild(td);
    batchTbodyEl.appendChild(row);
    return;
  }

  for (const item of items) {
    const row = document.createElement("tr");

    const similarity = Number(item.dblp_title_similarity);
    const similarityText = Number.isFinite(similarity) ? similarity.toFixed(4) : "-";
    const statusText = item.found ? t("status_matched") : t("status_not_matched");

    const cells = [
      String(item.index ?? ""),
      String(item.query_title ?? ""),
      statusText,
      String(item.dblp_title ?? "-"),
      similarityText,
      String(item.year ?? "-"),
      String(item.venue ?? "-"),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    batchTbodyEl.appendChild(row);
  }
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[,"\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function downloadBatchCsv() {
  if (!latestBatchItems.length) return;
  const headers = [
    "index",
    "query_title",
    "found",
    "dblp_id",
    "dblp_title",
    "dblp_title_similarity",
    "year",
    "venue",
    "pub_type",
    "duration_ms",
    "error_message",
  ];
  const lines = [headers.join(",")];
  for (const item of latestBatchItems) {
    const row = [
      item.index,
      item.query_title,
      item.found,
      item.dblp_id,
      item.dblp_title,
      item.dblp_title_similarity,
      item.year,
      item.venue,
      item.pub_type,
      item.duration_ms,
      item.error_message,
    ].map(csvEscape);
    lines.push(row.join(","));
  }
  const blob = new Blob([`\uFEFF${lines.join("\n")}`], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "citeverifier_batch_result.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function onSearch(event) {
  event.preventDefault();
  const title = normalizeTitle(titleInputEl.value);
  if (!title) {
    setMessage(t("msg_enter_title"), true);
    return;
  }

  const payload = {
    title,
    max_candidates: Number(maxCandidatesEl.value || "100000"),
  };

  setMessage(t("msg_searching"));
  try {
    const data = await fetchJson("/api/search/title", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    latestSingleResult = data;
    if (data.found) {
      renderMatch(data);
      setMessage(t("msg_match_found"));
    } else {
      renderNoMatch(data);
      setMessage(t("msg_no_match"));
    }
    await refreshHealth();
  } catch (err) {
    setMessage(t("msg_search_failed", { err: err.message }), true);
  }
}

async function extractTitlesFromFile(file) {
  if (!file) return [];
  const content = await file.text();
  const lowerName = (file.name || "").toLowerCase();
  if (lowerName.endsWith(".csv")) {
    const rows = content.split(/\r?\n/).map((line) => parseFirstCsvColumn(line));
    if (rows.length > 1 && /^title$/i.test(rows[0])) {
      rows.shift();
    }
    return rows.filter((line) => line.length > 0);
  }
  return parseLines(content);
}

async function onBatchSearch(event) {
  event.preventDefault();
  const textTitles = parseLines(batchInputEl.value);
  let fileTitles = [];
  if (batchFileEl.files && batchFileEl.files[0]) {
    try {
      fileTitles = await extractTitlesFromFile(batchFileEl.files[0]);
    } catch (err) {
      setBatchMessage(t("msg_file_read_failed", { err: err.message }), true);
      return;
    }
  }

  const titles = dedupeTitles([...textTitles, ...fileTitles]);
  if (titles.length === 0) {
    setBatchMessage(t("msg_batch_input_required"), true);
    return;
  }
  if (titles.length > MAX_BATCH_SIZE) {
    setBatchMessage(t("msg_batch_too_many", { max: MAX_BATCH_SIZE }), true);
    return;
  }

  const payload = {
    titles,
    max_candidates: Number(maxCandidatesEl.value || "100000"),
  };

  setBatchMessage(t("msg_batch_processing", { n: titles.length }));
  try {
    const data = await fetchJson("/api/search/title/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const summary = data.summary || null;
    const items = Array.isArray(data.items) ? data.items : [];
    latestBatchSummary = summary;
    latestBatchItems = items;
    renderBatchSummary(summary);
    renderBatchTable(items);
    batchDownloadBtnEl.disabled = items.length === 0;
    setBatchMessage(
      t("msg_batch_completed", { found: summary?.found_count ?? 0, total: items.length })
    );
    await refreshHealth();
  } catch (err) {
    latestBatchSummary = null;
    latestBatchItems = [];
    batchDownloadBtnEl.disabled = true;
    renderBatchSummary(null);
    renderBatchTable([]);
    setBatchMessage(t("msg_batch_failed", { err: err.message }), true);
  }
}

searchFormEl.addEventListener("submit", onSearch);
batchFormEl.addEventListener("submit", onBatchSearch);
batchDownloadBtnEl.addEventListener("click", downloadBatchCsv);

initLanguage();
renderBatchSummary(null);
renderBatchTable([]);
refreshHealth();

