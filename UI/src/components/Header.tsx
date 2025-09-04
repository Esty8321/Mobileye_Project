import React from "react";

export function Header() {
  return (
    <header className="bg-white/80 backdrop-blur sticky top-0 z-30 shadow-sm">
      <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-emerald-600 flex items-center justify-center text-white font-bold">
            AI
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-800 leading-tight">
              Image Pipeline
            </h1>
            <p className="text-xs text-gray-500">
              Classification → (Apple Health) → YOLO (filtered)
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="hidden sm:flex items-center gap-3 text-sm text-gray-500">
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />
            Healthy
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-red-500 inline-block" />
            Sick
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-4 h-0.5 bg-sky-500 inline-block" />
            YOLO Boxes
          </span>
        </div>
      </div>
    </header>
  );
}

