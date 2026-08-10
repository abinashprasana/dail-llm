import { useEffect, useId, useRef, useState } from "react";

interface MatrixCell {
  row: number;
  column: number;
  value: number;
}

const visuallyHidden = {
  position: "absolute",
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: "hidden",
  clip: "rect(0, 0, 0, 0)",
  whiteSpace: "nowrap",
  border: 0,
} as const;

function colorFor(value: number, maximum: number) {
  const t = maximum > 0 ? Math.min(1, Math.sqrt(value / maximum)) : 0;
  const start = [9, 29, 23];
  const middle = [44, 139, 104];
  const end = [233, 226, 208];
  const source = t < 0.68 ? start : middle;
  const target = t < 0.68 ? middle : end;
  const local = t < 0.68 ? t / 0.68 : (t - 0.68) / 0.32;
  return `rgb(${source.map((channel, index) => Math.round(channel + (target[index] - channel) * local)).join(",")})`;
}

function Heatmap({ matrix, labels, title }: { matrix: number[][]; labels: string[]; title: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const instructionsId = useId();
  const [pointerCell, setPointerCell] = useState<MatrixCell | null>(null);
  const [keyboardCell, setKeyboardCell] = useState<MatrixCell | null>(null);
  const rowCount = matrix.length;
  const columnCount = matrix.reduce((maximum, row) => Math.max(maximum, row.length), 0);
  const activeCell = pointerCell ?? keyboardCell;

  function getCell(row: number, column: number): MatrixCell | null {
    if (rowCount === 0) return null;
    const safeRow = Math.max(0, Math.min(rowCount - 1, row));
    const values = matrix[safeRow];
    if (!values?.length) return null;
    const safeColumn = Math.max(0, Math.min(values.length - 1, column));
    return { row: safeRow, column: safeColumn, value: values[safeColumn] };
  }

  useEffect(() => {
    const element = canvas.current;
    if (!element || matrix.length === 0) return;
    const size = 640;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    element.width = size * ratio;
    element.height = size * ratio;
    const context = element.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.fillStyle = "#091d17";
    context.fillRect(0, 0, size, size);
    const values = matrix.flat();
    const maximum = values.length > 0 ? Math.max(...values) : 0;
    const cellWidth = size / columnCount;
    const cellHeight = size / rowCount;
    matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      context.fillStyle = colorFor(value, maximum);
      context.fillRect(
        columnIndex * cellWidth,
        rowIndex * cellHeight,
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }));
  }, [columnCount, matrix, rowCount]);

  function handlePointer(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!canvas.current || rowCount === 0 || columnCount === 0) return;
    const bounds = canvas.current.getBoundingClientRect();
    const column = Math.max(0, Math.min(columnCount - 1, Math.floor(((event.clientX - bounds.left) / bounds.width) * columnCount)));
    const row = Math.max(0, Math.min(rowCount - 1, Math.floor(((event.clientY - bounds.top) / bounds.height) * rowCount)));
    const next = getCell(row, column);
    setPointerCell((current) => current?.row === next?.row && current?.column === next?.column ? current : next);
  }

  function handleKeyboard(event: React.KeyboardEvent<HTMLCanvasElement>) {
    const current = keyboardCell ?? getCell(0, 0);
    if (!current) return;
    let row = current.row;
    let column = current.column;

    if (event.key === "ArrowUp") row -= 1;
    else if (event.key === "ArrowDown") row += 1;
    else if (event.key === "ArrowLeft") column -= 1;
    else if (event.key === "ArrowRight") column += 1;
    else if (event.key === "Home") {
      row = 0;
      column = 0;
    } else if (event.key === "End") {
      row = rowCount - 1;
      column = columnCount - 1;
    } else return;

    event.preventDefault();
    setPointerCell(null);
    setKeyboardCell(getCell(row, column));
  }

  const queryLabel = activeCell ? (labels[activeCell.row] || `Character ${activeCell.row + 1}`) : "";
  const keyLabel = activeCell ? (labels[activeCell.column] || `Character ${activeCell.column + 1}`) : "";
  const keyboardStatus = pointerCell === null && keyboardCell !== null;

  return (
    <figure className="heatmap-figure">
      <span id={instructionsId} className="sr-only" style={visuallyHidden}>
        Focus the matrix and use the arrow keys to inspect query and key characters. Home moves to the first cell and End moves to the last.
      </span>
      <figcaption><span>{title}</span><small>Query ↓ · Key →</small></figcaption>
      <div className="heatmap-wrap">
        <canvas
          ref={canvas}
          onPointerMove={handlePointer}
          onPointerLeave={() => setPointerCell(null)}
          onFocus={() => setKeyboardCell((current) => current ?? getCell(0, 0))}
          onBlur={() => setKeyboardCell(null)}
          onKeyDown={handleKeyboard}
          tabIndex={rowCount > 0 && columnCount > 0 ? 0 : -1}
          role="img"
          aria-label={`${title} causal attention matrix with ${rowCount} query characters and ${columnCount} key characters`}
          aria-describedby={instructionsId}
        />
        {activeCell && (
          <span
            aria-hidden="true"
            style={{
              position: "absolute",
              left: `${(activeCell.column / columnCount) * 100}%`,
              top: `${(activeCell.row / rowCount) * 100}%`,
              width: `${100 / columnCount}%`,
              height: `${100 / rowCount}%`,
              boxShadow: "inset 0 0 0 1px var(--brass)",
              pointerEvents: "none",
            }}
          />
        )}
        {activeCell && (
          <div
            className="heatmap-tooltip"
            role={keyboardStatus ? "status" : undefined}
            aria-live={keyboardStatus ? "polite" : undefined}
            aria-atomic={keyboardStatus ? "true" : undefined}
          >
            <span>Query {queryLabel} · Key {keyLabel}</span>
            <strong>{activeCell.value.toFixed(4)}</strong>
          </div>
        )}
      </div>
    </figure>
  );
}

export function AttentionCanvas({ matrices, labels, layer, head }: {
  matrices: number[][][];
  labels: string[];
  layer: number;
  head: number | null;
}) {
  return (
    <div className={`heatmap-grid ${matrices.length > 1 ? "is-multiple" : ""}`}>
      {matrices.map((matrix, index) => (
        <Heatmap
          key={`${layer}-${head ?? "all"}-${index}`}
          matrix={matrix}
          labels={labels}
          title={`Layer ${layer + 1} · Head ${(head ?? index) + 1}`}
        />
      ))}
    </div>
  );
}
