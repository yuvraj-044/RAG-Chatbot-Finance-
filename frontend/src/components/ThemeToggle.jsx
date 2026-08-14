export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      <div className={`theme-toggle-knob ${theme}`}>
        {theme === "dark" ? "🌙" : "☀️"}
      </div>
    </button>
  );
}
