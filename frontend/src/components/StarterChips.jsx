const CHIPS = [
  {
    icon: "📊",
    text: "What was AAPL's quarterly revenue and net income?",
  },
  {
    icon: "🔍",
    text: "Show credit card fraud detection patterns",
  },
  {
    icon: "📈",
    text: "What are the latest NSE NIFTY 50 index trends?",
  },
  {
    icon: "🏦",
    text: "Compare the balance sheet of MSFT vs AAPL",
  },
];

export default function StarterChips({ onSelect }) {
  return (
    <div className="starter-section">
      <div className="starter-hero">
        <span className="starter-hero-icon">🏦</span>
        <h2>FinSight AI</h2>
        <p>
          Your AI-powered financial research assistant. Ask about quarterly
          financials, transaction fraud analysis, NSE market indices, and
          balance sheet comparisons — backed by real document citations.
        </p>
      </div>
      <div className="starter-chips">
        {CHIPS.map((chip, i) => (
          <button
            key={i}
            className="starter-chip"
            onClick={() => onSelect(chip.text)}
          >
            <span className="starter-chip-icon">{chip.icon}</span>
            {chip.text}
          </button>
        ))}
      </div>
    </div>
  );
}
