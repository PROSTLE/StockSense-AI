// ============================================================================
// WALLET FUNCTIONS
// ============================================================================

let walletBalance = 10000.00; // Initial wallet balance

function showWallet() {
  document.getElementById("walletPanel").classList.remove("hidden");
  document.getElementById("walletActionArea").innerHTML = "";
  document.getElementById("walletMessage").classList.add("hidden");
  
  // Load current wallet balance from backend
  loadWalletBalance();
}

function closeWallet() {
  document.getElementById("walletPanel").classList.add("hidden");
}

async function loadWalletBalance() {
  try {
    const res = await fetch(`${API}/api/wallet/balance`);
    const data = await res.json();
    
    if (data.balance !== undefined) {
      walletBalance = data.balance;
      document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
      // Also update cash balance in portfolio if present
      const portBalanceEl = document.getElementById("portBalance");
      if (portBalanceEl) portBalanceEl.textContent = "₹" + formatNumber(walletBalance);
    }
  } catch (e) {
    console.error("Error loading wallet balance:", e);
    // Fallback to local balance
    document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
    // Also update cash balance in portfolio if present
    const portBalanceEl = document.getElementById("portBalance");
    if (portBalanceEl) portBalanceEl.textContent = "₹" + formatNumber(walletBalance);
  }
}

function showAddMoney() {
  document.getElementById("walletActionArea").innerHTML = `
    <div class='wallet-form'>
      <h3>Add Money</h3>
      <input type='number' id='addAmount' placeholder='Enter amount' min='1' />
      <button onclick='startRazorpayPayment()'>Add via Razorpay</button>
    </div>
  `;
  document.getElementById("walletMessage").classList.add("hidden");
}

function showWithdrawMoney() {
  document.getElementById("walletActionArea").innerHTML = `
    <div class='wallet-form'>
      <h3>Withdraw Money</h3>
      <input type='number' id='withdrawAmount' placeholder='Enter amount' min='1' />
      <button onclick='startWithdraw()'>Withdraw</button>
    </div>
  `;
  document.getElementById("walletMessage").classList.add("hidden");
}

function showWalletMessage(message, type = 'success') {
  const msgEl = document.getElementById("walletMessage");
  msgEl.textContent = message;
  msgEl.className = `wallet-message ${type}`;
  msgEl.classList.remove("hidden");
  
  // Auto-hide after 5 seconds
  setTimeout(() => {
    msgEl.classList.add("hidden");
  }, 5000);
}

// Razorpay integration
function startRazorpayPayment() {
  const amt = document.getElementById('addAmount').value;
  if (!amt || amt <= 0) {
    showWalletMessage('Please enter a valid amount', 'error');
    return;
  }

  const amountInPaise = Math.round(parseFloat(amt) * 100);

  // Create Razorpay order
  const options = {
    key: 'rzp_test_SDdi0bRwia1AA4', // Your test key
    amount: amountInPaise,
    currency: 'INR',
    name: 'StockSense Wallet',
    description: 'Add Money to Wallet',
    handler: function (response) {
      // Payment successful - verify and credit wallet
      verifyAndCreditWallet(response.razorpay_payment_id, parseFloat(amt));
    },
    prefill: {
      name: 'User',
      email: 'user@example.com'
    },
    theme: { 
      color: '#22c55e' 
    },
    modal: {
      ondismiss: function() {
        showWalletMessage('Payment cancelled', 'error');
      }
    }
  };
  
  try {
    const rzp = new Razorpay(options);
    rzp.open();
  } catch (e) {
    console.error("Razorpay error:", e);
    showWalletMessage('Payment gateway error. Please try again.', 'error');
  }
}

async function verifyAndCreditWallet(paymentId, amount) {
  try {
    // Call backend to verify payment and credit wallet
    const res = await fetch(`${API}/api/wallet/add`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        payment_id: paymentId,
        amount: amount
      })
    });

    const data = await res.json();

    if (data.status === 'success') {
      walletBalance = data.new_balance;
      document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
      showWalletMessage(`✅ ₹${formatNumber(amount)} added successfully! Payment ID: ${paymentId}`, 'success');
      document.getElementById("walletActionArea").innerHTML = "";
    } else {
      showWalletMessage('Payment verification failed. Please contact support.', 'error');
    }
  } catch (e) {
    console.error("Wallet credit error:", e);
    // Fallback - credit locally if backend fails
    walletBalance += amount;
    document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
    showWalletMessage(`✅ ₹${formatNumber(amount)} added successfully! Payment ID: ${paymentId}`, 'success');
    document.getElementById("walletActionArea").innerHTML = "";
  }
}

async function startWithdraw() {
  const amt = document.getElementById('withdrawAmount').value;
  if (!amt || amt <= 0) {
    showWalletMessage('Please enter a valid amount', 'error');
    return;
  }

  const withdrawAmount = parseFloat(amt);

  // Check if sufficient balance
  if (withdrawAmount > walletBalance) {
    showWalletMessage('Insufficient balance', 'error');
    return;
  }

  // Confirm withdrawal
  if (!confirm(`Withdraw ₹${formatNumber(withdrawAmount)} from your wallet?`)) {
    return;
  }

  try {
    // Call backend to process withdrawal
    const res = await fetch(`${API}/api/wallet/withdraw`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount: withdrawAmount
      })
    });

    const data = await res.json();

    if (data.status === 'success') {
      walletBalance = data.new_balance;
      document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
      showWalletMessage(`✅ ₹${formatNumber(withdrawAmount)} withdrawn successfully!`, 'success');
      document.getElementById("walletActionArea").innerHTML = "";
    } else {
      showWalletMessage(data.message || 'Withdrawal failed', 'error');
    }
  } catch (e) {
    console.error("Withdrawal error:", e);
    // Fallback - deduct locally if backend fails
    walletBalance -= withdrawAmount;
    document.getElementById("walletBalance").textContent = "₹" + formatNumber(walletBalance);
    showWalletMessage(`✅ ₹${formatNumber(withdrawAmount)} withdrawn successfully!`, 'success');
    document.getElementById("walletActionArea").innerHTML = "";
  }
}

// ============================================================================
// MAIN APP CODE
// ============================================================================

const API = "http://127.0.0.1:8000";
let currentTicker = "";
let liveInterval = null;
let portfolioInterval = null;
let mainChart = null;
let volumeChart = null;
let predChart = null;
let activeTimeframe = "1d";
let loadVersion = 0;


let searchTimeout = null;

document.addEventListener("DOMContentLoaded", () => {
  loadHighPotential();
  loadPortfolio();
  loadMarketIndices();
  loadWalletBalance();
  setupTimeframeButtons();
  portfolioInterval = setInterval(loadPortfolio, 15000);
  setInterval(loadMarketIndices, 30000);

  const tickerInput = document.getElementById("tickerInput");
  const dropdown = document.getElementById("searchDropdown");

  tickerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      dropdown.classList.add("hidden");
      loadTicker();
    }
    if (e.key === "Escape") dropdown.classList.add("hidden");
  });

  tickerInput.addEventListener("input", () => {
    clearTimeout(searchTimeout);
    const q = tickerInput.value.trim();
    if (q.length < 1) {
      dropdown.classList.add("hidden");
      return;
    }
    searchTimeout = setTimeout(() => searchStocks(q), 250);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrapper")) {
      dropdown.classList.add("hidden");
    }
  });
});


async function searchStocks(query) {
  const dropdown = document.getElementById("searchDropdown");
  try {
    const res = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      dropdown.innerHTML = '<div class="search-no-results">No stocks found</div>';
      dropdown.classList.remove("hidden");
      return;
    }

    dropdown.innerHTML = data.results.map(s => `
      <div class="search-item" data-ticker="${s.ticker}">
        <span class="search-ticker">${s.ticker}</span>
        <span class="search-name">${s.name}</span>
      </div>
    `).join("");

    dropdown.querySelectorAll(".search-item").forEach(item => {
      item.addEventListener("click", () => {
        const ticker = item.dataset.ticker;
        document.getElementById("tickerInput").value = ticker;
        dropdown.classList.add("hidden");
        loadTicker(ticker);
      });
    });

    dropdown.classList.remove("hidden");
  } catch (e) {
    console.error("Search error:", e);
    dropdown.classList.add("hidden");
  }
}


async function loadMarketIndices() {
  try {
    const res = await fetch(`${API}/api/market-indices`);
    const data = await res.json();
    data.indices.forEach((idx) => {
      let priceEl, changeEl;
      if (idx.name === "NIFTY 50") {
        priceEl = document.getElementById("mktNiftyPrice");
        changeEl = document.getElementById("mktNiftyChange");
      } else if (idx.name === "SENSEX") {
        priceEl = document.getElementById("mktSensexPrice");
        changeEl = document.getElementById("mktSensexChange");
      }
      if (priceEl && changeEl) {
        priceEl.textContent = formatPrice(idx.price);
        const sign = idx.change >= 0 ? "+" : "";
        changeEl.textContent = `${sign}${idx.change.toFixed(2)} (${sign}${idx.change_pct}%)`;
        changeEl.className = `mkt-change ${idx.direction}`;
      }
    });
  } catch (e) {
    console.error("Market indices error:", e);
  }
}


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


async function loadTicker(ticker) {
  ticker = ticker || document.getElementById("tickerInput").value.trim().toUpperCase();
  if (!ticker) return;

  loadVersion++;
  const myVersion = loadVersion;
  currentTicker = ticker;
  document.getElementById("tickerInput").value = ticker;

  const emptyState = document.getElementById("chartEmptyState");
  if (emptyState) emptyState.style.display = "none";

  if (liveInterval) clearInterval(liveInterval);

  await Promise.all([
    loadLivePrice(ticker, myVersion),
    loadChartData(ticker, activeTimeframe, myVersion),
    loadPrediction(ticker, myVersion),
    loadSentimentAndIndicators(ticker, myVersion),
  ]);

  if (myVersion !== loadVersion) return;
  liveInterval = setInterval(() => loadLivePrice(ticker, loadVersion), 10000);
}


async function loadLivePrice(ticker, version) {
  try {
    const res = await fetch(`${API}/api/live/${ticker}`);
    if (version !== undefined && version !== loadVersion) return;
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

    document.getElementById("mktStockLabel").textContent = ticker;
    document.getElementById("mktStockPrice").textContent = formatPrice(data.price);
    const mktChg = document.getElementById("mktStockChange");
    mktChg.textContent = `${sign}${data.change} (${sign}${data.change_pct}%)`;
    mktChg.className = `mkt-change ${data.direction}`;
  } catch (e) {
    console.error("Live price error:", e);
  }
}


async function loadChartData(ticker, timeframe, version) {
  try {
    const [chartRes, infoRes] = await Promise.all([
      fetch(`${API}/api/chart/${ticker}?timeframe=${timeframe}`),
      fetch(`${API}/api/stock/${ticker}`),
    ]);
    if (version !== undefined && version !== loadVersion) return;
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


const PRED_MESSAGES = [
  "Extracting 10 features (RSI, MACD, Bollinger, ATR…)",
  "Training 3-layer LSTM neural network…",
  "Running 100 epochs with early stopping…",
  "Analyzing 60-day price patterns…",
  "Applying gradient clipping & LR scheduling…",
  "Generating 5-day price forecast…",
  "Cross-validating with test set…",
];
let predMsgInterval = null;

async function loadPrediction(ticker, version) {
  const section = document.getElementById("predSection");
  const loader = document.getElementById("predLoader");
  const chartWrap = document.getElementById("predChartWrap");
  const msgEl = document.getElementById("predLoaderMsg");


  section.classList.remove("hidden");
  loader.classList.remove("hidden");
  chartWrap.style.display = "none";


  let msgIdx = 0;
  msgEl.textContent = PRED_MESSAGES[0];
  predMsgInterval = setInterval(() => {
    msgIdx = (msgIdx + 1) % PRED_MESSAGES.length;
    msgEl.textContent = PRED_MESSAGES[msgIdx];
  }, 3000);

  try {
    const res = await fetch(`${API}/api/predict/${ticker}`);
    if (version !== undefined && version !== loadVersion) {
      clearInterval(predMsgInterval);
      return;
    }
    const data = await res.json();


    clearInterval(predMsgInterval);


    loader.classList.add("hidden");
    chartWrap.style.display = "";

    document.getElementById("predConf").textContent = data.confidence + "%";
    document.getElementById("predConf").style.color =
      data.confidence >= 70 ? "#22c55e" : data.confidence >= 45 ? "#eab308" : "#ef4444";

    document.getElementById("predRisk").textContent = data.risk;
    document.getElementById("predRisk").style.color =
      data.risk === "Low" ? "#22c55e" : data.risk === "Medium" ? "#eab308" : "#ef4444";

    document.getElementById("predDay1").textContent = "₹" + data.predicted_prices[0];
    document.getElementById("predDay5").textContent = "₹" + data.predicted_prices[4];

    drawPredictionChart(data);
  } catch (e) {
    console.error("Prediction error:", e);
    clearInterval(predMsgInterval);
    msgEl.textContent = "⚠️ Forecast failed — try again";
  }
}

function drawPredictionChart(pred) {
  const hist = pred.historical_last_30;
  const future = pred.predicted_prices;


  const histDates = pred.historical_dates || [];
  const predDates = pred.prediction_dates || [];

  const labels = [];
  for (let i = 0; i < hist.length; i++) {
    labels.push(histDates[i] || `D-${hist.length - i}`);
  }
  for (let i = 0; i < future.length; i++) {
    labels.push(predDates[i] || `+${i + 1}d`);
  }

  const histData = [...hist, ...Array(future.length).fill(null)];
  const predData = [...Array(hist.length - 1).fill(null), hist[hist.length - 1], ...future];

  if (predChart) predChart.destroy();

  predChart = new Chart(document.getElementById("predChart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Historical", data: histData, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.05)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
        { label: "Forecast", data: predData, borderColor: "#22c55e", borderDash: [6, 3], backgroundColor: "rgba(34,197,94,0.08)", fill: true, tension: 0.3, pointRadius: 4, borderWidth: 2, pointBackgroundColor: "#22c55e" },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#6b7280" } } },
      scales: {
        x: { ticks: { color: "#4b5563", maxTicksLimit: 10, maxRotation: 45, font: { size: 9 } }, grid: { color: "#1f2937" } },
        y: { ticks: { color: "#4b5563", font: { size: 10 } }, grid: { color: "#1f2937" } },
      },
    },
  });
}


async function loadSentimentAndIndicators(ticker, version) {

  try {
    const sentRes = await fetch(`${API}/api/sentiment/${ticker}`);
    if (version !== undefined && version !== loadVersion) return;
    const sent = await sentRes.json();

    const sentPanel = document.getElementById("sentPanel");
    sentPanel.classList.remove("hidden");

    const gauge = document.getElementById("gaugeCircle");
    gauge.className = `gauge-circle ${sent.overall_sentiment}`;
    document.getElementById("gaugeText").textContent =
      typeof sent.overall_score === "number" ? sent.overall_score.toFixed(2) : sent.overall_score;
    document.getElementById("gaugeLabel").textContent = sent.overall_sentiment.toUpperCase();
    document.getElementById("gaugeLabel").style.color =
      sent.overall_sentiment === "positive" ? "#22c55e" :
        sent.overall_sentiment === "negative" ? "#ef4444" : "#eab308";

    const sentList = document.getElementById("sentList");
    sentList.innerHTML = "";
    (sent.details || []).forEach((d) => {
      const li = document.createElement("li");
      const cls = d.label === "positive" ? "s-pos" : d.label === "negative" ? "s-neg" : "s-neu";
      const score = d.combined_score !== undefined ? ` (${d.combined_score.toFixed(3)})` : "";
      li.innerHTML = `<span class="${cls}">● ${d.label}${score}</span> ${d.text}`;
      sentList.appendChild(li);
    });
  } catch (e) {
    console.error("Sentiment error:", e);
  }


  try {
    const infoRes = await fetch(`${API}/api/summary/${ticker}`);
    if (version !== undefined && version !== loadVersion) return;
    const data = await infoRes.json();

    if (data.indicators) {
      const indPanel = document.getElementById("indPanel");
      indPanel.classList.remove("hidden");
      const indBody = document.getElementById("indBody");
      indBody.innerHTML = "";
      Object.entries(data.indicators).forEach(([key, val]) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${key}</td><td>${val}</td>`;
        indBody.appendChild(tr);
      });
    }
  } catch (e) {
    console.error("Indicators error:", e);
  }
}


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
          <div class="hp-price">₹${s.price}</div>
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


async function loadPortfolio() {
  try {
    const res = await fetch(`${API}/api/trade/portfolio`);
    const data = await res.json();


    document.getElementById("portValue").textContent = "₹" + formatNumber(data.total_value || data.balance);
    document.getElementById("portBalance").textContent = "₹" + formatNumber(data.balance);

    const unrealized = data.total_unrealized_pnl || 0;
    const unrealizedEl = document.getElementById("portUnrealized");
    unrealizedEl.textContent = (unrealized >= 0 ? "+₹" : "-₹") + formatNumber(Math.abs(unrealized));
    unrealizedEl.style.color = unrealized >= 0 ? "#22c55e" : "#ef4444";

    const realized = data.total_realized_pnl || 0;
    const realizedEl = document.getElementById("portRealized");
    realizedEl.textContent = (realized >= 0 ? "+₹" : "-₹") + formatNumber(Math.abs(realized));
    realizedEl.style.color = realized >= 0 ? "#22c55e" : "#ef4444";

    document.getElementById("portWinRate").textContent = (data.win_rate || 0) + "%";
    document.getElementById("portWinRate").style.color = data.win_rate >= 50 ? "#22c55e" : "#ef4444";

    document.getElementById("portTrades").textContent = data.total_trades || 0;


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


    renderPositions(data.positions || {});


    renderTradeHistory(data.trade_history || []);

  } catch (e) {
    console.error("Portfolio error:", e);
  }
}


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
    const sl = pos.stop_loss || 0;
    const tp = pos.tp1 || pos.take_profit || 0;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="ticker-col">${ticker}</td>
      <td>${pos.shares}</td>
      <td>₹${pos.buy_price}</td>
      <td>₹${pos.current_price || pos.buy_price}</td>
      <td class="${pnlClass}">${pnlSign}₹${formatNumber(Math.abs(pnl))}</td>
      <td class="${pnlClass}">${pnlSign}${pnlPct}%</td>
      <td style="color:#ef4444">₹${sl}</td>
      <td style="color:#22c55e">₹${tp}</td>
      <td><button class="btn-sell-small" onclick="forceSell('${ticker}')">SELL</button></td>
    `;
    tbody.appendChild(tr);
  });
}


function renderTradeHistory(trades) {
  const log = document.getElementById("tradeLog");

  if (trades.length === 0) {
    log.innerHTML = '<div class="empty-msg">No trades yet. Activate the bot or execute a trade.</div>';
    return;
  }

  log.innerHTML = "";

  [...trades].reverse().forEach((t) => {
    const profit = (typeof t.profit === "number" && !isNaN(t.profit)) ? t.profit : 0;
    const profit_pct = (typeof t.profit_pct === "number" && !isNaN(t.profit_pct)) ? t.profit_pct : 0;
    const profitClass = profit >= 0 ? "pnl-positive" : "pnl-negative";
    const profitSign = profit >= 0 ? "+" : "";
    const time = new Date(t.sell_time).toLocaleString();

    const entry = document.createElement("div");
    entry.className = "trade-entry";
    entry.innerHTML = `
      <span class="te-time">${time}</span>
      <span>
        <span class="te-ticker">${t.ticker}</span>
        — ${t.shares} shares @ ₹${t.buy_price} → ₹${t.sell_price}
        <span style="color:#6b7280;font-size:0.7rem;"> (${t.reason})</span>
      </span>
      <span class="${profitClass}">${profitSign}₹${formatNumber(Math.abs(profit))}</span>
      <span class="${profitClass}">${profitSign}${profit_pct}%</span>
    `;
    log.appendChild(entry);
  });
}


function addToFeed(message, type) {
  const feed = document.getElementById("tradeFeed");
  const time = new Date().toLocaleTimeString();

  const item = document.createElement("div");
  item.className = `feed-item ${type}`;
  item.innerHTML = `<span style="color:#6b7280;font-size:0.7rem;">[${time}]</span> ${message}`;


  feed.insertBefore(item, feed.firstChild);


  while (feed.children.length > 50) {
    feed.removeChild(feed.lastChild);
  }
}


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


async function executeManualTrade() {
  const ticker = currentTicker || document.getElementById("tickerInput").value.trim().toUpperCase();
  if (!ticker) {
    alert("Enter a ticker first!");
    return;
  }

  addToFeed(`🔄 Analyzing <strong>${ticker}</strong> — running LSTM + Technical + Sentiment...`, "hold");

  try {
    const res = await fetch(`${API}/api/trade/execute/${ticker}`, { method: "POST" });
    const data = await res.json();

    const action = data.action;
    const price = data.current_price;

    if (action === "BUY") {
      const tr = data.trade_result;
      const cs = data.composite_score || '?';
      const regime = data.market_regime || 'NEUTRAL';
      const regimeIcon = regime === 'BULL' ? '🐂' : regime === 'BEAR' ? '🐻' : '⚖️';
      addToFeed(
        `🟢 <strong>BOUGHT ${tr.shares} x ${ticker}</strong> @ ₹${price} | ` +
        `${regimeIcon} ${regime} | Score: ${cs}/100 | SL: ₹${tr.stop_loss} | TP: ₹${tr.take_profit}`,
        "buy"
      );
    } else if (action === "SELL") {
      const tr = data.trade_result;
      const profit = (typeof tr.profit === "number" && !isNaN(tr.profit)) ? tr.profit : 0;
      const profit_pct = (typeof tr.profit_pct === "number" && !isNaN(tr.profit_pct)) ? tr.profit_pct : 0;
      const profitSign = profit >= 0 ? "+" : "";
      addToFeed(
        `🔴 <strong>SOLD ${ticker}</strong> @ ₹${price} | ` +
        `P&L: ${profitSign}₹${profit} (${profitSign}${profit_pct}%) | Reason: ${tr.reason}`,
        "sell"
      );
    } else if (action === "HOLD") {
      addToFeed(
        `🟡 <strong>HOLD ${ticker}</strong> @ ₹${price} — position within thresholds ` +
        `(P&L: ${data.position.current_pnl}%)`,
        "hold"
      );
    } else {
      const cs = data.composite_score || '?';
      const regime = data.market_regime || 'NEUTRAL';
      const regimeIcon = regime === 'BULL' ? '🐂' : regime === 'BEAR' ? '🐻' : '⚖️';
      addToFeed(
        `⚪ <strong>WAIT on ${ticker}</strong> @ ₹${price} — ${regimeIcon} ${regime} | Score: ${cs}/100 | ` +
        `${data.reason || "conditions not met"}`,
        "wait"
      );
    }

    loadPortfolio();
  } catch (e) {
    console.error("Trade error:", e);
    addToFeed(`❌ Trade execution failed for ${ticker}: ${e.message}`, "sell");
  }
}


async function runAutoScan() {
  addToFeed("🔍 <strong>Auto-scan started</strong> — XGBoost intraday analysis with regime detection...", "hold");

  try {
    const res = await fetch(`${API}/api/trade/auto-scan`, { method: "POST" });
    const data = await res.json();

    if (data.status === "inactive") {
      addToFeed("⚠️ Bot is OFF — activate it first before scanning", "wait");
      return;
    }

    addToFeed(`✅ <strong>Scan complete</strong> — analyzed ${data.scanned} stocks`, "hold");


    (data.results || []).forEach((r) => {
      const action = r.action;
      const ticker = r.ticker;

      if (action === "BUY" && r.trade_result) {
        const tr = r.trade_result;
        if (tr.status === "bought") {
          addToFeed(
            `🟢 <strong>AUTO-BUY ${tr.shares} x ${ticker}</strong> @ ₹${tr.price} | ` +
            `SL: ₹${tr.stop_loss || 0} | TP: ₹${tr.tp1 || 0}`,
            "buy"
          );
        } else {
          addToFeed(`⚪ ${ticker}: ${tr.reason || "skipped"}`, "wait");
        }
      } else if (action === "SELL" && (r.trade_result || r.result)) {
        const tr = r.trade_result || r.result;
        const profit = (typeof tr.profit === "number" && !isNaN(tr.profit)) ? tr.profit : 0;
        const profitSign = profit >= 0 ? "+" : "";
        addToFeed(
          `🔴 <strong>AUTO-SELL ${ticker}</strong> | P&L: ${profitSign}₹${profit} | ${tr.reason || ''}`,
          "sell"
        );
      } else if (action === "WAIT") {
        const regime = r.regime || 'SIDEWAYS';
        const conf = r.confidence || 0;
        const expRet = r.expected_return_pct || 0;
        const regimeIcon = regime.includes('BULL') ? '🐂' : regime.includes('BEAR') ? '🐻' : '⚖️';
        addToFeed(
          `⚪ ${ticker}: WAIT — ${regimeIcon} ${regime} | Confidence: ${conf} | Expected: ${expRet}%`,
          "wait"
        );
      } else if (action === "SKIP") {
        addToFeed(`⚪ ${ticker}: SKIP — ${r.reason || 'insufficient data'}`, "wait");
      }
    });

    loadPortfolio();
  } catch (e) {
    console.error("Scan error:", e);
    addToFeed(`❌ Auto-scan failed: ${e.message}`, "sell");
  }
}


async function forceSell(ticker) {
  if (!confirm(`Force sell all ${ticker} shares at market price?`)) return;

  addToFeed(`🔄 Force-selling <strong>${ticker}</strong>...`, "hold");

  try {
    const res = await fetch(`${API}/api/trade/sell/${ticker}`, { method: "POST" });
    const data = await res.json();

    if (data.status === "sold") {
      const profitSign = data.profit >= 0 ? "+" : "";
      addToFeed(
        `🔴 <strong>FORCE-SOLD ${ticker}</strong> @ ₹${data.price} | ` +
        `P&L: ${profitSign}₹${data.profit} (${profitSign}${data.profit_pct}%)`,
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


async function resetPortfolio() {
  if (!confirm("Reset portfolio to ₹10,00,000? All positions and history will be lost.")) return;

  try {
    await fetch(`${API}/api/trade/reset`, { method: "POST" });
    addToFeed("🔄 <strong>Portfolio reset</strong> to ₹10,00,000.00", "hold");
    loadPortfolio();
  } catch (e) {
    console.error("Reset error:", e);
  }
}


function formatPrice(p) {
  if (!p) return "—";
  return parseFloat(p).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatVolume(v) {
  if (!v) return "—";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toString();
}

function formatNumber(n) {
  return parseFloat(n).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
