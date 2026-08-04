import type { AgentStepDetail } from '@/types/agentSteps'

export function StepDetailCard({ detail }: { detail: AgentStepDetail }) {
  const hasBody = Boolean(
    detail.query ||
      detail.error ||
      detail.display_text ||
      (detail.columns?.length && detail.rows?.length),
  )
  if (!hasBody) return null

  return (
    <div className="mt-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 px-3 py-2.5 space-y-2">
      {detail.query && (
        <div>
          <p className="text-[11px] text-slate-400 dark:text-slate-500 mb-1">查询</p>
          <p className="text-xs text-slate-700 dark:text-slate-200 leading-relaxed">
            {detail.query}
          </p>
        </div>
      )}
      {detail.error && (
        <div>
          <p className="text-[11px] text-rose-500 dark:text-rose-400 mb-1">错误</p>
          <p className="text-xs text-rose-600 dark:text-rose-300 leading-relaxed">
            {detail.error}
          </p>
        </div>
      )}
      {detail.display_text && (
        <div>
          <p className="text-[11px] text-slate-400 dark:text-slate-500 mb-1">摘要</p>
          <p className="text-xs text-slate-700 dark:text-slate-200 leading-relaxed">
            {detail.display_text}
          </p>
        </div>
      )}
      {detail.columns && detail.rows && detail.rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs text-left">
            <thead>
              <tr className="text-slate-400 dark:text-slate-500">
                {detail.columns.map((column) => (
                  <th key={column} className="pr-3 pb-1 font-medium whitespace-nowrap">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((row, rowIndex) => (
                <tr
                  key={`row-${rowIndex}`}
                  className="text-slate-700 dark:text-slate-200 border-t border-slate-200/80 dark:border-slate-700/80"
                >
                  {row.map((cell, cellIndex) => (
                    <td
                      key={`cell-${rowIndex}-${cellIndex}`}
                      className="pr-3 py-1 whitespace-nowrap"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
