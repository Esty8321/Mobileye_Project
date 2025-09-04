import React from "react";

export function Features() {
  return (
    <section className="grid md:grid-cols-3 gap-6">
      {/* Step 1: Classification */}
      <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
        <div className="text-emerald-600 font-semibold text-sm mb-2">
          Step 1 — General Classification
        </div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Identify the main class in the image
        </h3>
        <p className="text-gray-600 text-sm leading-6">
          The general model returns a <b>label</b> (e.g., <code>apple</code>,{" "}
          <code>car</code>, …) and a <b>score</b>. The UI displays the class
          name and its probability.
        </p>
      </div>

      {/* Step 2: Apple Health */}
      <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
        <div className="text-emerald-600 font-semibold text-sm mb-2">
          Step 2 — Apple Health (conditional)
        </div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Healthy / Sick for <code>apple</code>
        </h3>
        <p className="text-gray-600 text-sm leading-6">
          If the predicted class is <code>apple</code>, the Apple Health model
          runs. The result is shown with a color tag:{" "}
          <span className="text-emerald-600 font-medium">Green = Healthy</span>,{" "}
          <span className="text-red-600 font-medium">Red = Sick</span>. If the
          model is not available, the UI shows a note to train it.
        </p>
      </div>

      {/* Step 3: YOLO */}
      <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
        <div className="text-emerald-600 font-semibold text-sm mb-2">
          Step 3 — YOLO Detections
        </div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Only for the class from Step 1
        </h3>
        <p className="text-gray-600 text-sm leading-6">
          YOLO returns detection boxes for all objects. The server already{" "}
          <b>filters</b> them to the class predicted in Step 1. The UI draws
          boxes with a <span className="font-medium text-sky-600">blue</span>{" "}
          outline using normalized coordinates (<code>box_norm</code>) so they
          scale with the image.
        </p>
      </div>
    </section>
  );
}
