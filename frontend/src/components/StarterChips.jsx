const CHIPS = [
  {
    label: "Revenue",
    text: "What was AAPL's quarterly revenue and net income?",
  },
  {
    label: "Fraud",
    text: "Show credit card fraud detection patterns",
  },
  {
    label: "Market",
    text: "What are the latest NSE NIFTY 50 index trends?",
  },
  {
    label: "Compare",
    text: "Compare the balance sheet of MSFT vs AAPL",
  },
];

export default function StarterChips({ onSelect }) {
  return (
    <div className="starter-section">
      <div className="starter-hero">
        <span className="starter-eyebrow">Grounded financial research</span>
        <h2>Ask sharp questions. Get cited answers.</h2>
        <p>
          FinSight combines your RAG pipeline with a focused analyst-style
          workspace for filings, ratios, transaction patterns, market indices,
          and document-backed comparisons.
        </p>
      </div>
      <div className="starter-chips">
        {CHIPS.map((chip, i) => (
          <button
            key={i}
            className="starter-chip"
            onClick={() => onSelect(chip.text)}
          >
            <span className="starter-chip-label">{chip.label}</span>
            <span>{chip.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
