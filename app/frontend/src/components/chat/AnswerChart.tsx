import type { AnswerChartSpec } from '@/types/api'

function maxAbs(values: number[]): number {
  return values.reduce((peak, value) => Math.max(peak, Math.abs(value)), 0) || 1
}

export function AnswerChart({ chart }: { chart: AnswerChartSpec }) {
  const categories = chart.categories ?? []
  const barSeries = chart.bars?.[0]
  const lineSeries = chart.lines?.[0]
  const barValues = barSeries?.values ?? []
  const lineValues = lineSeries?.values ?? []
  if (categories.length < 2) return null

  const width = 360
  const height = 160
  const padX = 28
  const padTop = 16
  const padBottom = 28
  const plotW = width - padX * 2
  const plotH = height - padTop - padBottom
  const n = categories.length
  const slot = plotW / n
  const barPeak = maxAbs(barValues)
  const linePeak = maxAbs(lineValues)
  const zeroY = padTop + plotH / 2

  const barRects = barValues.map((value, index) => {
    const x = padX + slot * index + slot * 0.25
    const h = (Math.abs(value) / barPeak) * (plotH * 0.42)
    const y = value >= 0 ? zeroY - h : zeroY
    return { x, y, w: slot * 0.5, h, value }
  })

  const linePoints = lineValues
    .map((value, index) => {
      const x = padX + slot * index + slot / 2
      const y = zeroY - (value / linePeak) * (plotH * 0.42)
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="mt-3 mb-1 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-3">
      {chart.title && (
        <p className="text-xs font-medium text-slate-600 dark:text-slate-300 mb-2">
          {chart.title}
          {chart.unit ? `（${chart.unit}）` : ''}
        </p>
      )}
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-md">
        <line
          x1={padX}
          y1={zeroY}
          x2={width - padX}
          y2={zeroY}
          stroke="currentColor"
          className="text-slate-200 dark:text-slate-700"
          strokeWidth="1"
        />
        {barRects.map((rect, index) => (
          <g key={`bar-${index}`}>
            <rect
              x={rect.x}
              y={rect.y}
              width={rect.w}
              height={Math.max(rect.h, 1)}
              rx="3"
              className="fill-sky-500/80"
            />
            <text
              x={rect.x + rect.w / 2}
              y={rect.value >= 0 ? rect.y - 4 : rect.y + rect.h + 11}
              textAnchor="middle"
              className="fill-slate-500 text-[9px]"
            >
              {Number.isFinite(rect.value)
                ? Math.abs(rect.value) >= 100
                  ? rect.value.toFixed(0)
                  : rect.value.toFixed(1)
                : ''}
            </text>
          </g>
        ))}
        {linePoints && (
          <>
            <polyline
              points={linePoints}
              fill="none"
              strokeWidth="2"
              className="stroke-orange-500"
            />
            {lineValues.map((value, index) => {
              const x = padX + slot * index + slot / 2
              const y = zeroY - (value / linePeak) * (plotH * 0.42)
              return (
                <circle
                  key={`pt-${index}`}
                  cx={x}
                  cy={y}
                  r="3"
                  className="fill-orange-500"
                />
              )
            })}
          </>
        )}
        {categories.map((label, index) => (
          <text
            key={`cat-${label}-${index}`}
            x={padX + slot * index + slot / 2}
            y={height - 8}
            textAnchor="middle"
            className="fill-slate-400 text-[9px]"
          >
            {label}
          </text>
        ))}
      </svg>
      <div className="mt-1 flex gap-3 text-[10px] text-slate-400">
        {barSeries && (
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm bg-sky-500/80" />
            {barSeries.name}
          </span>
        )}
        {lineSeries && (
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-500" />
            {lineSeries.name}
          </span>
        )}
      </div>
    </div>
  )
}
