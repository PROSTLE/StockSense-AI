/* ══════════════════════════════════════════════════════
   StockSense AI — Frontend v3.0 (COMPLETE FILE)
   Live Charts + Prediction + Auto Trading Bot
   ══════════════════════════════════════════════════════ */

const API = "http://127.0.0.1:8000";
let currentTicker = "";
let liveInterval = null;
let portfolioInterval = null;
let mainChart = null;
let volumeChart = null;
let predChart = null;
let activeTimeframe = "1d";

/* ══ INIT ══════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  loadHighPotential();
  loadPortfolio();
  setupTimeframeButtons();
  portfolioInterval = setInterval(loadPortfolio, 15000);

  document.getElementById("tickerInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadTicker();
  });
});

/* ══ TAB SWITCHING ═════════════════════════════════════ */
function switchTab(tab) {
  document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach((t) => t.classList.add("hidden"));

  if (tab === "charts") {
    document.getElementById("chartsTab").classList.remove("hidden");
    document.querySelectorAll(".nav-tab")[0].classList.add("active");
  } else {
    document.getElementById("tradingTab").classList.remove("hidden");
    document.querySelectorAll(".nav-tab")[1].classList.add("active");
    loadPortfolio();
  }
}

/* ══ TIMEFRAME BUTTONS ═════════════════════════════════ */
function setupTimeframeButtons() {
  document.querySelectorAll(".tf-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tf-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTimeframe = btn.dataset.tf;
      if (currentTicker) loadChartData(currentTicker, activeTimeframe);
    });
  });
}

/* ══ MAIN LOAD TICKER ══════════════════════════════════ */
async function loadTicker(ticker) {
  ticker = ticker || document.getElementById("tickerInput").value.trim().toUpperCase();
  if (!ticker) return;

  currentTicker = ticker;
  document.getElementById("tickerInput").value = ticker;

  if (liveInterval) clearInterval(liveInterval);

  await Promise.all([
    loadLivePrice(ticker),
    loadChartData(ticker, activeTimeframe),
    loadPrediction(ticker),
    loadSentimentAndIndicators(ticker),
  ]);

  liveInterval = setInterval(() => loadLivePrice(ticker), 10000);
}

/* ══ LIVE PRICE ════════════════════════════════════════ */
async function loadLivePrice(ticker) {
  try {
    const res = await fetch(`${API}/api/live/${ticker}`);
    const data = await res.json();

    document.getElementById("priceBanner").classList.remove("hidden");
    document.getElementById("bannerName").textContent = ticker;
    document.getElementById("bannerPrice").textContent = formatPrice(data.price);

    const changeEl = document.getElementById("bannerChange");
    const sign = data.change >= 0 ? "+" : "";
    changeEl.textContent = `${sign}${data.change} (${sign}${data.change_pct}%)`;
    changeEl.className = `price-change ${data.direction}`;

    document.getElementById("statHigh").textContent = formatPrice(data.day_high);
    document.getElementById("statLow").textContent = formatPrice(data.day_low);
    document.getElementById("statVol").textContent = formatVolume(data.volume);
  } catch (e) {
    console.error("Live price error:", e);
  }
}

/* ══ CHART DATA ════════════════════════════════════════ */
async function loadChartData(ticker, timeframe) {
  try {
    const [chartRes, infoRes] = await Promise.all([
      fetch(`${API}/api/chart/${ticker}?timeframe=${timeframe}`),
      fetch(`${API}/api/stock/${ticker}`),
    ]);
    const chartData = await chartRes.json();
    const stockData = await infoRes.json();

    if (stockData.info) {
      const info = stockData.info;
      document.getElementById("bannerName").textContent = `${info.name} (${ticker})`;
      document.getElementById("statOpen").textContent = formatPrice(info.open);
      document.getElementById("statHigh").textContent = formatPrice(info.day_high);
      document.getElementById("statLow").textContent = formatPrice(info.day_low);
      document.getElementById("statVol").textContent = formatVolume(info.volume);
      document.getElementById("stat52H").textContent = formatPrice(info.fifty_two_week_high);
      document.getElementById("stat52L").textContent = formatPrice(info.fifty_two_week_low);
    }

    drawMainChart(chartData.data);
    drawVolumeChart(chartData.data);
  } catch (e) {
    console.error("Chart error:", e);
  }
}

/* ══ DRAW MAIN CHART ═══════════════════════════════════ */
function drawMainChart(data) {
  const labels = data.map((d) => d.time);
  const closes = data.map((d) => d.close);
  const highs = data.map((d) => d.high);
  const lows = data.map((d) => d.low);

  const isUp = closes[closes.length - 1] >= closes[0];
  const lineColor = isUp ? "#22c55e" : "#ef4444";
  const fillColor = isUp ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)";

  if (mainChart) mainChart.destroy();

  mainChart = new Chart(document.getElementById("mainChart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Close", data: closes, borderColor: lineColor, backgroundColor: fillColor, fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
        { label: "High", data: highs, borderColor: "rgba(34,197,94,0.2)", borderDash: [2, 4], pointRadius: 0, borderWidth: 1, fill: false },
        { label: "Low", data: lows, borderColor: "rgba(239,68,68,0.2)", borderDash: [2, 4], pointRadius: 0, borderWidth: 1, fill: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#6b7280", font: { size: 11 } } },
        tooltip: { backgroundColor: "#1a1f2e", borderColor: "#242a38", borderWidth: 1, titleColor: "#fff", bodyColor: "#d1d5db" },
      },
      scales: {
        x: { ticks: { color: "#4b5563", maxTicksLimit: 10, maxRotation: 0, font: { size: 10 } }, grid: { color: "#1f2937" } },
        y: { ticks: { color: "#4b5563", font: { size: 10 } }, grid: { color: "#1f2937" } },
      },
    },
  });
}

/* ══ DRAW VOLUME CHART ═════════════════════════════════ */
function drawVolumeChart(data) {
  const labels = data.map((d) => d.time);
  const volumes = data.map((d) => d.volume);
  const colors = data.map((d, i) => {
    if (i === 0) return "rgba(59,130,246,0.5)";
    return d.close >= data[i - 1].close ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)";
  });

  if (volumeChart) volumeChart.destroy();

  volumeChart = new Chart(document.getElementById("volumeChart").getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Volume", data: volumes, backgroundColor: colors, borderWidth: 0 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { ticks: { color: "#4b5563", font: { size: 9 } }, grid: { color: "#1f2937" } },
      },
    },
  });
}

/* ══ PREDICTION ════════════════════════════════════════ */
async function loadPrediction(ticker) {
  const section = document.getElementById("predSection");
  try {
    const res = await fetch(`${API}/api/predict/${ticker}`);
    const data = await res.json();

    section.classList.remove("hidden");

    document.getElementById("predConf").textContent = data.confidence + "%";
    document.getElementById("predConf").style.color =
      data.confidence >= 70 ? "#22c55e" : data.confidence >= 45 ? "#eab308" : "#ef4444";

    document.getElementById("predRisk").textContent = data.risk;
    document.getElementById("predRisk").style.color =
      data.risk === "Low" ? "#22c55e" : data.risk === "Medium" ? "#eab308" : "#ef4444";

    document.getElementById("predDay1").textContent = "$" + data.predicted_prices[0];
    document.getElementById("predDay5").textContent = "$" + data.predicted_prices[4];

    drawPredictionChart(data);
  } catch (e) {
    console.error("Prediction error:", e);
    section.classList.add("hidden");
  }
}

function drawPredictionChart(pred) {
  const hist = pred.historical_last_30;
  const future = pred.predicted_prices;

  const labels = [];
  for (let i = 0; i < hist.length; i++) labels.push(`D-${hist.length - i}`);
  for (let i = 1; i <= future.length; i++) labels.push(`+${i}d`);

  const histData = [...hist, ...Array(future.length).fill(null)];
  const predData = [...Array(hist.length - 1).fill(null), hist[hist.length - 1], ...future];

  if (predChart) predChart.destroy();

  predChart = new Chart(document.getElementById("predChart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Historical", data: histData, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.05)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
        { label: "Predicted", data: predData, borderColor: "#22c55e", borderDash: [6, 3], backgroundColor: "rgba(34,197,94,0.08)", fill: true, tension: 0.3, pointRadius: 4, borderWidth: 2, pointBackgroundColor: "#22c55e" },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#6b7280" } } },
      scales: {
        x: { ticks: { color: "#4b5563", maxTicksLimit: 12, font: { size: 10 } }, grid: { color: "#1f2937" } },
        y: { ticks: { color: "#4b5563", font: { size: 10 } }, grid: { color: "#1f2937" } },
      },
    },
  });
}

/* ══ SENTIMENT + INDICATORS ════════════════════════════ */
async function loadSentimentAndIndicators(ticker) {
  try {
    const res = await fetch(`${API}/api/summary/${ticker}`);
    const data = await res.json();

    // Sentiment
    const sentPanel = document.getElementById("sentPanel");
    sentPanel.classList.remove("hidden");

    const sent = data.sentiment;
    const gauge = document.getElementById("gaugeCircle");
    gauge.className = `gauge-circle ${sent.overall_sentiment}`;
    document.getElementById("gaugeText").textContent = sent.overall_score;
    document.getElementById("gaugeLabel").textContent = sent.overall_sentiment.toUpperCase();
    document.getElementById("gaugeLabel").style.color =
      sent.overall_sentiment === "positive" ? "#22c55e" :
      sent.overall_sentiment === "negative" ? "#ef4444" : "#eab308";

    const sentList = document.getElementById("sentList");
    sentList.innerHTML = "";
    (sent.details || []).forEach((d) => {
      const li = document.createElement("li");
      const cls = d.label === "positive" ? "s-pos" : d.label === "negative" ? "s-neg" : "s-neu";
      li.innerHTML = `<span class="${cls}">● ${d.label}</span> ${d.text}`;
      sentList.appendChild(li);
    });

    // Indicators
    const indPanel = document.getElementById("indPanel");
    indPanel.classList.remove("hidden");
    const indBody = document.getElementById("indBody");
    indBody.innerHTML = "";
    Object.entries(data.indicators).forEach(([key, val]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${key}</td><td>${val}</td>`;
      indBody.appendChild(tr);
    });
  } catch (e) {
    console.error("Summary error:", e);
  }
}

/* ══ HIGH POTENTIAL STOCKS ═════════════════════════════ */
async function loadHighPotential() {
  try {
    const res = await fetch(`${API}/api/high-potential`);
    const data = await res.json();

    document.getElementById("hpLoader").classList.add("hidden");
    const list = document.getElementById("hpList");
    list.innerHTML = "";

    data.stocks.forEach((s) => {
      const div = document.createElement("div");
      div.className = "hp-item";
      div.onclick = () => {
        document.getElementById("tickerInput").value = s.ticker;
        loadTicker(s.ticker);
      };

      const changeClass = s.change_pct >= 0 ? "up" : "down";
      const sign = s.change_pct >= 0 ? "+" : "";
      const scoreClass = s.score >= 70 ? "high" : s.score >= 45 ? "med" : "low";

      div.innerHTML = `
        <div>
          <div class="hp-ticker">${s.ticker}</div>
          <div class="hp-name">${s.name}</div>
        </div>
        <div>
          <div class="hp-price">$${s.price}</div>
          <div class="hp-change ${changeClass}">${sign}${s.change_pct}%</div>
        </div>
        <div class="hp-score ${scoreClass}">${s.signal}</div>
      `;
      list.appendChild(div);
    });
  } catch (e) {
    document.getElementById("hpLoader").textContent = "⚠️ Scanner unavailable";
    console.error("High potential error:", e);
  }
}

/* ══════════════════════════════════════════════════════
   TRADING FUNCTIONS
   ══════════════════════════════════════════════════════ */

/* ══ LOAD PORTFOLIO ════════════════════════════════════ */
async function loadPortfolio() {
  try {
    const res = await fetch(`${API}/api/trade/portfolio`);
    const data = await res.json();

    // Portfolio header cards
    document.getElementById("portValue").textContent = "$" + formatNumber(data.total_value || data.balance);
    document.getElementById("portBalance").textContent = "$" + formatNumber(data.balance);

    const unrealized = data.total_unrealized_pnl || 0;
    const unrealizedEl = document.getElementById("portUnrealized");
    unrealizedEl.textContent = (unrealized >= 0 ? "+$" : "-$") + formatNumber(Math.abs(unrealized));
    unrealizedEl.style.color = unrealized >= 0 ? "#22c55e" : "#ef4444";

    const realized = data.total_realized_pnl || 0;
    const realizedEl = document.getElementById("portRealized");
    realizedEl.textContent = (realized >= 0 ? "+$" : "-$") + formatNumber(Math.abs(realized));
    realizedEl.style.color = realized >= 0 ? "#22c55e" : "#ef4444";

    document.getElementById("portWinRate").textContent = (data.win_rate || 0) + "%";
    document.getElementById("portWinRate").style.color = data.win_rate >= 50 ? "#22c55e" : "#ef4444";

    document.getElementById("portTrades").textContent = data.total_trades || 0;

    // Bot status
    const botDot = document.getElementById("botDot");
    const botStatus = document.getElementById("botStatus");
    const toggleBtn = document.getElementById("toggleBotBtn");

    if (data.bot_active) {
      botDot.className = "bot-dot active";
      botStatus.textContent = "Bot: ACTIVE";
      botStatus.style.color = "#22c55e";
      toggleBtn.textContent = "⏹ Deactivate Bot";
      toggleBtn.classList.add("active");
    } else {
      botDot.className = "bot-dot";
      botStatus.textContent = "Bot: OFF";
      botStatus.style.color = "#ef4444";
      toggleBtn.textContent = "⚡ Activate Bot";
      toggleBtn.classList.remove("active");
    }

    // Open positions table
    renderPositions(data.positions || {});

    // Trade history
    renderTradeHistory(data.trade_history || []);

  } catch (e) {
    console.error("Portfolio error:", e);
  }
}

/* ══ RENDER POSITIONS TABLE ════════════════════════════ */
function renderPositions(positions) {
  const tbody = document.getElementById("positionsBody");

  if (Object.keys(positions).length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-msg">No open positions — activate the bot or trade manually</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  Object.entries(positions).forEach(([ticker, pos]) => {
    const pnl = pos.unrealized_pnl || 0;
    const pnlPct = pos.pnl_pct || 0;
    const pnlClass = pnl >= 0 ? "pnl-positive" : "pnl-negative";
    const pnlSign = pnl >= 0 ? "+" : "";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="ticker-col">${ticker}</td>
      <td>${pos.shares}</td>
      <td>$${pos.buy_price}</td>
      <td>$${pos.current_price || pos.buy_price}</td>
      <td class="${pnlClass}">${pnlSign}$${formatNumber(Math.abs(pnl))}</td>
      <td class="${pnlClass}">${pnlSign}${pnlPct}%</td>
      <td style="color:#ef4444">$${pos.stop_loss}</td>
      <td style="color:#22c55e">$${pos.take_profit}</td>
      <td><button class="btn-sell-small" onclick="forceSell('${ticker}')">SELL</button></td>
    `;
    tbody.appendChild(tr);
  });
}

/* ══ RENDER TRADE HISTORY ══════════════════════════════ */
function renderTradeHistory(trades) {
  const log = document.getElementById("tradeLog");

  if (trades.length === 0) {
    log.innerHTML = '<div class="empty-msg">No trades yet. Activate the bot or execute a trade.</div>';
    return;
  }

  log.innerHTML = "";
  // Show most recent first
  [...trades].reverse().forEach((t) => {
    const profitClass = t.profit >= 0 ? "pnl-positive" : "pnl-negative";
    const profitSign = t.profit >= 0 ? "+" : "";
    const time = new Date(t.sell_time).toLocaleString();

    const entry = document.createElement("div");
    entry.className = "trade-entry";
    entry.innerHTML = `
      <span class="te-time">${time}</span>
      <span>
        <span class="te-ticker">${t.ticker}</span>
        — ${t.shares} shares @ $${t.buy_price} → $${t.sell_price}
        <span style="color:#6b7280;font-size:0.7rem;"> (${t.reason})</span>
      </span>
      <span class="${profitClass}">${profitSign}$${formatNumber(Math.abs(t.profit))}</span>
      <span class="${profitClass}">${profitSign}${t.profit_pct}%</span>
    `;
    log.appendChild(entry);
  });
}

/* ══ ADD TO TRADE FEED ═════════════════════════════════ */
function addToFeed(message, type) {
  const feed = document.getElementById("tradeFeed");
  const time = new Date().toLocaleTimeString();

  const item = document.createElement("div");
  item.className = `feed-item ${type}`;
  item.innerHTML = `<span style="color:#6b7280;font-size:0.7rem;">[${time}]</span> ${message}`;

  // Add to top
  feed.insertBefore(item, feed.firstChild);

  // Keep only last 50 entries
  while (feed.children.length > 50) {
    feed.removeChild(feed.lastChild);
  }
}

/* ══ TOGGLE BOT ════════════════════════════════════════ */
async function toggleBot() {
  try {
    const res = await fetch(`${API}/api/trade/toggle`, { method: "POST" });
    const data = await res.json();

    if (data.bot_active) {
      addToFeed("🤖 <strong>Auto-trading bot ACTIVATED</strong> — scanning for opportunities...", "buy");
    } else {
      addToFeed("⏹ <strong>Auto-trading bot DEACTIVATED</strong>", "sell");
    }

    loadPortfolio();
  } catch (e) {
    console.error("Toggle error:", e);
    addToFeed("❌ Failed to toggle bot", "sell");
  }
}

/* ══ EXECUTE MANUAL TRADE ══════════════════════════════ */
async function executeManualTrade() {
  const ticker = currentTicker || document.getElementById("tickerInput").value.trim().toUpperCase();
  if (!ticker) {
    alert("Enter a ticker first!");
    return;
  }

  addToFeed(`🔄 Analyzing <strong>${ticker}</strong> — running LSTM prediction...`, "hold");

  try {
    const res = await fetch(`${API}/api/trade/execute/${ticker}`, { method: "POST" });
    const data = await res.json();

    const action = data.action;
    const price = data.current_price;

    if (action === "BUY") {
      const tr = data.trade_result;
      addToFeed(
        `🟢 <strong>BOUGHT ${tr.shares} x ${ticker}</strong> @ $${price} | ` +
        `SL: $${tr.stop_loss} | TP: $${tr.take_profit} | Cost: $${tr.cost}`,
        "buy"
      );
    } else if (action === "SELL") {
      const tr = data.trade_result;
      const profitSign = tr.profit >= 0 ? "+" : "";
      addToFeed(
        `🔴 <strong>SOLD ${ticker}</strong> @ $${price} | ` +
        `P&L: ${profitSign}$${tr.profit} (${profitSign}${tr.profit_pct}%) | Reason: ${tr.reason}`,
        "sell"
      );
    } else if (action === "HOLD") {
      addToFeed(
        `🟡 <strong>HOLD ${ticker}</strong> @ $${price} — position within thresholds ` +
        `(P&L: ${data.position.current_pnl}%)`,
        "hold"
      );
    } else {
      addToFeed(
        `⚪ <strong>WAIT on ${ticker}</strong> @ $${price} — ${data.reason || "conditions not met for entry"} ` +
        `(Pred: ${data.pred_direction} ${data.pred_change_pct}%, Conf: ${data.confidence}%)`,
        "wait"
      );
    }

    loadPortfolio();
  } catch (e) {
    console.error("Trade error:", e);
    addToFeed(`❌ Trade execution failed for ${ticker}: ${e.message}`, "sell");
  }
}

/* ══ AUTO SCAN & TRADE ═════════════════════════════════ */
async function runAutoScan() {
  addToFeed("🔍 <strong>Auto-scan started</strong> — scanning 8 stocks with LSTM...", "hold");

  try {
    const res = await fetch(`${API}/api/trade/auto-scan`, { method: "POST" });
    const data = await res.json();

    if (data.status === "inactive") {
      addToFeed("⚠️ Bot is OFF — activate it first before scanning", "wait");
      return;
    }

    addToFeed(`✅ <strong>Scan complete</strong> — analyzed ${data.scanned} stocks`, "hold");

    // Log each result
    (data.results || []).forEach((r) => {
      const action = r.action;
      const ticker = r.ticker;

      if (action === "BUY" && r.trade_result) {
        const tr = r.trade_result;
        if (tr.status === "bought") {
          addToFeed(
            `🟢 <strong>AUTO-BUY ${tr.shares} x ${ticker}</strong> @ $${tr.price} | ` +
            `SL: $${tr.stop_loss} | TP: $${tr.take_profit}`,
            "buy"
          );
        } else {
          addToFeed(`⚪ ${ticker}: ${tr.reason || "skipped"}`, "wait");
        }
      } else if (action === "SELL" && r.trade_result) {
        const tr = r.trade_result || r.result;
        const profitSign = tr.profit >= 0 ? "+" : "";
        addToFeed(
          `🔴 <strong>AUTO-SELL ${ticker}</strong> | P&L: ${profitSign}$${tr.profit} | ${tr.reason}`,
          "sell"
        );
      } else if (action === "WAIT") {
        addToFeed(
          `⚪ ${ticker}: WAIT (Pred: ${r.pred_direction} ${r.pred_change_pct}%, Conf: ${r.confidence}%)`,
          "wait"
        );
      }
    });

    loadPortfolio();
  } catch (e) {
    console.error("Scan error:", e);
    addToFeed(`❌ Auto-scan failed: ${e.message}`, "sell");
  }
}

/* ══ FORCE SELL ════════════════════════════════════════ */
async function forceSell(ticker) {
  if (!confirm(`Force sell all ${ticker} shares at market price?`)) return;

  addToFeed(`🔄 Force-selling <strong>${ticker}</strong>...`, "hold");

  try {
    const res = await fetch(`${API}/api/trade/sell/${ticker}`, { method: "POST" });
    const data = await res.json();

    if (data.status === "sold") {
      const profitSign = data.profit >= 0 ? "+" : "";
      addToFeed(
        `🔴 <strong>FORCE-SOLD ${ticker}</strong> @ $${data.price} | ` +
        `P&L: ${profitSign}$${data.profit} (${profitSign}${data.profit_pct}%)`,
        "sell"
      );
    } else {
      addToFeed(`⚠️ ${ticker}: ${data.reason}`, "wait");
    }

    loadPortfolio();
  } catch (e) {
    console.error("Force sell error:", e);
    addToFeed(`❌ Force sell failed: ${e.message}`, "sell");
  }
}

/* ══ RESET PORTFOLIO ═══════════════════════════════════ */
async function resetPortfolio() {
  if (!confirm("Reset portfolio to $100,000? All positions and history will be lost.")) return;

  try {
    await fetch(`${API}/api/trade/reset`, { method: "POST" });
    addToFeed("🔄 <strong>Portfolio reset</strong> to $100,000.00", "hold");
    loadPortfolio();
  } catch (e) {
    console.error("Reset error:", e);
  }
}

/* ══ HELPERS ═══════════════════════════════════════════ */
function formatPrice(p) {
  if (!p) return "—";
  return parseFloat(p).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatVolume(v) {
  if (!v) return "—";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toString();
}

function formatNumber(n) {
  return parseFloat(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
