// ── Mock API Layer ──────────────────────────────────────────────────────
// Returns realistic finance responses matching the actual project datasets:
//   - LQ_*.csv, Hist_*.csv          → Financial statements
//   - credit_card_transactions*.csv → Fraud detection
//   - nse_indexes.csv               → NSE market data
//   - peers_comparisons_data.csv    → Corporate comparisons

const MOCK_DELAY_MIN = 600;
const MOCK_DELAY_MAX = 1800;

const MOCK_RESPONSES = {
  quarterly_revenue: {
    reply:
      "Based on the quarterly financial data from **LQ_AAPL_Q3_2024.csv**, here are AAPL's key financials:\n\n| Metric | Value |\n|--------|-------|\n| **Total Revenue** | $85,777M |\n| **Net Income** | $21,448M |\n| **EPS (Diluted)** | $1.40 |\n| **Gross Margin** | 46.3% |\n| **Operating Cash Flow** | $29,051M |\n\nRevenue grew 5% year-over-year, driven primarily by the Services segment which hit a record $24.2B. iPhone revenue contributed $46.2B (+2.3% YoY).\n\n[SOURCE: LQ_AAPL_Q3_2024.csv | Ticker: AAPL | Date: 2024-09-30]",
    sources: [
      {
        doc_title: "LQ_AAPL_Q3_2024.csv",
        chunk_text:
          "[Financial Record]\nSource : LQ_AAPL_Q3_2024.csv | Ticker: AAPL\nPeriod : 2024-09-30 | Category: quarterly_financials\n  Total Revenue: 85777000000\n  Net Income: 21448000000\n  Eps Diluted: 1.40\n  Gross Margin: 0.463\n  Operating Cash Flow: 29051000000\n  Services Revenue: 24213000000",
        score: 0.96,
        ticker: "AAPL",
        date: "2024-09-30",
      },
      {
        doc_title: "Hist_Income_Stmt_AAPL.csv",
        chunk_text:
          "[Financial Record]\nSource : Hist_Income_Stmt_AAPL.csv | Ticker: AAPL\nPeriod : 2024-06-30 | Category: historical_financials\n  Total Revenue: 90753000000\n  Net Income: 23636000000\n  Revenue Growth Yoy: 0.049\n  Operating Margin: 0.301",
        score: 0.88,
        ticker: "AAPL",
        date: "2024-06-30",
      },
    ],
    latency_ms: 142,
    is_grounded: true,
  },
  fraud_detection: {
    reply:
      "Based on the **credit card transaction and fraud detection datasets**, here are the key fraud patterns identified:\n\n### Fraud Detection Summary\n- **Total Transactions Analyzed**: 284,807\n- **Fraudulent Transactions**: 492 (0.17% of total)\n- **Average Fraud Amount**: $122.21\n- **Average Legitimate Amount**: $88.29\n\n### Key Fraud Patterns\n1. **High-value anomalies**: Fraud transactions average 38% higher amounts than legitimate ones\n2. **Time clustering**: 68% of fraudulent transactions occur between 11 PM – 4 AM\n3. **Merchant category**: Gas stations and online retailers show highest fraud rates\n4. **Geographic spread**: Cross-state transactions are 3.2x more likely to be fraudulent\n\n### Risk Indicators\n- Transactions above $500 with new merchants → **High risk**\n- Multiple transactions within 10 minutes → **Medium risk**\n- First-time international transaction → **Medium risk**\n\n[SOURCE: credit_card_transactions.csv | fraud_detection_bank.csv]",
    sources: [
      {
        doc_title: "credit_card_transactions.csv",
        chunk_text:
          "[Transaction Record]\nSource : credit_card_transactions.csv\nDate   : 2024-06-15 | Category: transaction_record\n  Trans Num: txn_892341\n  Amount: 347.89\n  Merchant: Online Retailer\n  Category: shopping_net\n  Is Fraud: Yes\n  City: Springfield\n  State: IL",
        score: 0.93,
        ticker: "N/A",
        date: "2024-06-15",
      },
      {
        doc_title: "fraud_detection_bank.csv",
        chunk_text:
          "[Transaction Record]\nSource : fraud_detection_bank.csv\nDate   : 2024-07-02 | Category: transaction_record\n  Transaction Id: fd_10234\n  Amount: 892.50\n  Is Fraud: Yes\n  Risk Score: 0.94\n  Channel: online\n  Time: 02:34:00\n  Fraud Type: identity_theft",
        score: 0.89,
        ticker: "N/A",
        date: "2024-07-02",
      },
      {
        doc_title: "sd254_users.csv",
        chunk_text:
          "[Transaction Record]\nSource : sd254_users.csv\nDate   : N/A | Category: transaction_record\n  Customer Id: user_254\n  Credit Limit: 15000\n  Current Balance: 12340\n  Num Cards: 3\n  Account Age Months: 18\n  Risk Category: high",
        score: 0.81,
        ticker: "N/A",
        date: "N/A",
      },
    ],
    latency_ms: 218,
    is_grounded: true,
  },
  nse_market: {
    reply:
      "Here are the latest **NSE NIFTY 50 index trends** from the indexed market data:\n\n### NIFTY 50 Performance\n| Metric | Value |\n|--------|-------|\n| **Latest Close** | 25,010.90 |\n| **YTD Return** | +16.2% |\n| **52-Week High** | 25,078 |\n| **52-Week Low** | 18,837 |\n| **P/E Ratio** | 23.4x |\n| **Dividend Yield** | 1.23% |\n\n### Recent Trends\n• NIFTY 50 crossed **25,000 for the first time** in August 2024\n• **FII flows** turned net positive in July (+₹12,400 Cr) after 3 months of selling\n• **Banking sector (NIFTY BANK)** outperformed with +18.5% YTD returns\n• **IT sector (NIFTY IT)** lagged at +8.3% due to delayed US rate cuts\n• **VIX India** at 12.8, indicating low volatility environment\n\n[SOURCE: nse_indexes.csv | indexes_df.csv]",
    sources: [
      {
        doc_title: "nse_indexes.csv",
        chunk_text:
          "[Market Index Record]\nSource : nse_indexes.csv | Symbol: NIFTY 50\nDate   : 2024-08-01\n  Index Name: NIFTY 50\n  Open: 24892.15\n  High: 25078.30\n  Low: 24856.70\n  Close: 25010.90\n  Volume: 342567800\n  Change: 0.47\n  Change Pct: 1.62",
        score: 0.95,
        ticker: "NIFTY50",
        date: "2024-08-01",
      },
      {
        doc_title: "indexes_df.csv",
        chunk_text:
          "[Market Index Record]\nSource : indexes_df.csv | Symbol: NIFTY BANK\nDate   : 2024-08-01\n  Index Name: NIFTY BANK\n  Open: 51234.50\n  High: 51890.20\n  Low: 51100.80\n  Close: 51678.40\n  Ytd Return: 0.185\n  Pe Ratio: 15.8",
        score: 0.87,
        ticker: "NIFTYBANK",
        date: "2024-08-01",
      },
    ],
    latency_ms: 156,
    is_grounded: true,
  },
  balance_sheet_compare: {
    reply:
      "Here's a **balance sheet comparison** between **Microsoft (MSFT)** and **Apple (AAPL)** from the historical financial statements:\n\n| Metric | MSFT | AAPL |\n|--------|------|------|\n| **Total Assets** | $411.98B | $352.58B |\n| **Total Liabilities** | $205.75B | $290.44B |\n| **Shareholders' Equity** | $206.23B | $62.15B |\n| **Cash & Equivalents** | $75.53B | $29.97B |\n| **Total Debt** | $47.03B | $111.09B |\n| **Debt-to-Equity** | 0.23x | 1.79x |\n| **Current Ratio** | 1.77 | 0.99 |\n| **Book Value/Share** | $27.68 | $4.03 |\n\n### Key Insights\n• **MSFT has a stronger balance sheet** with D/E of 0.23 vs AAPL's 1.79\n• **AAPL relies heavily on debt financing** — $111B total debt vs $62B equity\n• **MSFT holds 3x more cash** ($75.5B vs $30B), providing more financial flexibility\n• **AAPL's negative working capital** (current ratio < 1) is typical for its business model but contrasts with MSFT's conservative approach\n\n[SOURCE: Hist_BS_Fin_Stmt.csv | Statement_Analysis_MSFT.csv]",
    sources: [
      {
        doc_title: "Hist_BS_Fin_Stmt.csv",
        chunk_text:
          "[Financial Record]\nSource : Hist_BS_Fin_Stmt.csv | Ticker: MSFT\nPeriod : 2024-06-30 | Category: historical_financials\n  Total Assets: 411976000000\n  Total Liabilities: 205753000000\n  Shareholders Equity: 206223000000\n  Cash And Equivalents: 75530000000\n  Total Debt: 47032000000\n  Current Assets: 159726000000\n  Current Liabilities: 90154000000",
        score: 0.94,
        ticker: "MSFT",
        date: "2024-06-30",
      },
      {
        doc_title: "Hist_BS_Fin_Stmt.csv",
        chunk_text:
          "[Financial Record]\nSource : Hist_BS_Fin_Stmt.csv | Ticker: AAPL\nPeriod : 2024-06-30 | Category: historical_financials\n  Total Assets: 352583000000\n  Total Liabilities: 290437000000\n  Shareholders Equity: 62146000000\n  Cash And Equivalents: 29965000000\n  Total Debt: 111088000000\n  Current Assets: 128757000000\n  Current Liabilities: 129651000000",
        score: 0.91,
        ticker: "AAPL",
        date: "2024-06-30",
      },
    ],
    latency_ms: 195,
    is_grounded: true,
  },
  out_of_scope: {
    reply:
      "I couldn't find relevant financial documents to answer that question. My knowledge base covers:\n\n• **Quarterly financials** — revenue, income, EPS from LQ_*.csv files (AAPL, MSFT, etc.)\n• **Historical balance sheets** — assets, liabilities, equity from Hist_*.csv files\n• **Credit card fraud detection** — transaction patterns from credit_card_transactions.csv\n• **NSE market indices** — NIFTY 50, NIFTY BANK from nse_indexes.csv\n• **Corporate comparisons** — peer analysis from peers_comparisons_data.csv\n\nTry asking about specific tickers, financial metrics, fraud patterns, or market indices.",
    sources: [],
    latency_ms: 89,
    is_grounded: false,
  },
};

function matchResponse(query) {
  const q = query.toLowerCase();
  if (q.includes("revenue") || q.includes("quarterly") || q.includes("net income") || q.includes("eps") || q.includes("earnings"))
    return MOCK_RESPONSES.quarterly_revenue;
  if (q.includes("fraud") || q.includes("credit card") || q.includes("transaction") || q.includes("suspicious"))
    return MOCK_RESPONSES.fraud_detection;
  if (q.includes("nse") || q.includes("nifty") || q.includes("index") || q.includes("market trend"))
    return MOCK_RESPONSES.nse_market;
  if (q.includes("balance sheet") || q.includes("compare") || q.includes("assets") || q.includes("debt") || q.includes("equity") || q.includes("vs"))
    return MOCK_RESPONSES.balance_sheet_compare;
  return MOCK_RESPONSES.out_of_scope;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Simulate random failures (5% chance)
function shouldFail() {
  return Math.random() < 0.05;
}

export async function mockChat(sessionId, message) {
  const start = performance.now();
  const waitMs =
    MOCK_DELAY_MIN + Math.random() * (MOCK_DELAY_MAX - MOCK_DELAY_MIN);
  await delay(waitMs);

  if (shouldFail()) {
    throw new Error("Service temporarily unavailable. The AI model is experiencing high traffic.");
  }

  const response = matchResponse(message);
  const elapsed = Math.round(performance.now() - start);

  return {
    session_id: sessionId,
    reply: response.reply,
    sources: response.sources,
    latency_ms: response.latency_ms || elapsed,
    is_grounded: response.is_grounded,
  };
}

export async function mockHealth() {
  await delay(100);
  return { status: "ok", stub_mode: true };
}

export async function mockReset(sessionId) {
  await delay(100);
  return { session_id: sessionId, status: "reset" };
}
