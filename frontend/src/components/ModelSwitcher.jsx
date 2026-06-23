import "./ModelSwitcher.css";

const OPTIONS = [
  {
    value: "hybrid",
    label: "Hybrid",
    desc: "RoBERTa for dates & law · Gemini for parties & clauses",
    dot: "dot-hybrid",
  },
  {
    value: "roberta",
    label: "RoBERTa only",
    desc: "Extractive QA · finds exact spans · runs fully offline",
    dot: "dot-roberta",
  },
  {
    value: "gemini",
    label: "Gemini only",
    desc: "Generative · summarises answers · needs API key",
    dot: "dot-gemini",
  },
];

export default function ModelSwitcher({ value, onChange }) {
  return (
    <div className="switcher-wrap">
      <div className="switcher-label">Extraction Model</div>
      <div className="switcher-options">
        {OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={`switcher-btn ${value === opt.value ? "active" : ""}`}
            onClick={() => onChange(opt.value)}
            type="button"
          >
            <span className={`switcher-dot ${opt.dot}`} />
            <span className="switcher-btn-label">{opt.label}</span>
            <span className="switcher-btn-desc">{opt.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
