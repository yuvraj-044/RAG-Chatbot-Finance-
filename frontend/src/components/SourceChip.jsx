export default function SourceChip({ count, onClick }) {
  if (!count || count === 0) return null;

  return (
    <button className="source-chip" onClick={onClick}>
      Sources ({count})
    </button>
  );
}
