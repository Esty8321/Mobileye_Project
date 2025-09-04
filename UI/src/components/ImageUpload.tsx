import { useEffect, useMemo, useState } from "react";
import { predictPipeline, type PipelineResponse, annotateImage } from "../api";

export default function ImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PipelineResponse | null>(null);

  // separate flags so "Annotate" doesn't block "Predict"
  const [predicting, setPredicting] = useState(false);
  const [saving, setSaving] = useState(false);

  const [labelForAnnotate, setLabelForAnnotate] = useState("");

  function onFileChange(f: File | null) {
    setFile(f);
    setResult(null);
    // reset flags to ensure buttons are enabled
    setPredicting(false);
    setSaving(false);

    // manage preview URL
    if (!f) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  }

  // cleanup preview URL when component unmounts or file changes
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function onPredict() {
    if (!file || predicting) return;
    setPredicting(true);
    setResult(null);
    try {
      const res = await predictPipeline(file);
      setResult(res);
    } catch (e) {
      console.error(e);
      alert("Prediction failed. Check backend and Network tab.");
    } finally {
      setPredicting(false); // always re-enable
    }
  }

  async function onAnnotate() {
    if (!file || !labelForAnnotate.trim() || saving) return;
    setSaving(true);
    try {
      const res = await annotateImage(file, labelForAnnotate.trim());
      if (res.ok) {
        alert("Annotation saved!");
        setLabelForAnnotate("");
      } else {
        alert("Annotate failed: " + (res.error || "unknown error"));
      }
    } catch (e) {
      console.error(e);
      alert("Annotate request failed.");
    } finally {
      setSaving(false);
    }
  }

  // frame color logic for apple health
  const frameClass = useMemo(() => {
    if (!result) return "border-gray-300";
    if (result.classification.label !== "apple" || !result.apple_health) {
      return "border-gray-300";
    }
    return result.apple_health.label === "healthy"
      ? "border-emerald-500"
      : "border-red-500";
  }, [result]);

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-bold">Image Pipeline Demo</h1>

      <div className="flex items-center gap-3">
        <label className="px-3 py-2 bg-emerald-600 text-white rounded cursor-pointer">
          Choose file
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => onFileChange(e.target.files?.[0] || null)}
          />
        </label>

        <button
          onClick={onPredict}
          disabled={!file || predicting}
          className={`px-4 py-2 rounded ${
            !file || predicting
              ? "bg-gray-300 cursor-not-allowed text-gray-600"
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {predicting ? "Predicting..." : "Predict"}
        </button>
      </div>

      {previewUrl && (
        <div className="mt-3">
          {/* Image + overlay for YOLO boxes */}
          <div className={`relative inline-block rounded-lg border-4 ${frameClass}`}>
            <img
              src={previewUrl}
              alt="preview"
              className="block max-w-full h-auto rounded-lg"
              style={{ maxHeight: 480 }}
            />

            {/* YOLO boxes (normalized) */}
            {result?.yolo?.map((d, i) => {
              const [x1, y1, x2, y2] = d.box_norm; // 0..1
              return (
                <div
                  key={i}
                  title={`${d.label} ${(d.score * 100).toFixed(1)}%`}
                  className="absolute"
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                    border: "2px solid #0af",
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.6) inset",
                    pointerEvents: "none",
                  }}
                />
              );
            })}
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <input
          value={labelForAnnotate}
          onChange={(e) => setLabelForAnnotate(e.target.value)}
          placeholder="Annotate label (optional)"
          className="flex-1 border rounded px-3"
        />
        <button
          onClick={onAnnotate}
          disabled={!file || !labelForAnnotate.trim() || saving}
          className={`px-4 py-2 rounded ${
            !file || !labelForAnnotate.trim() || saving
              ? "bg-gray-300 cursor-not-allowed text-gray-600"
              : "bg-emerald-600 text-white hover:bg-emerald-700"
          }`}
        >
          {saving ? "Saving..." : "Annotate"}
        </button>
      </div>

      {/* results */}
      {result && (
        <div className="mt-3 text-sm text-gray-800 space-y-2">
          <div>
            <span className="font-semibold">Class:</span>{" "}
            {result.classification.label}{" "}
            <span className="text-gray-500">
              ({(result.classification.score * 100).toFixed(1)}%)
            </span>
          </div>

          {result.classification.label === "apple" && (
            <div>
              <span className="font-semibold">Apple health:</span>{" "}
              {result.apple_health
                ? `${result.apple_health.label} (${(result.apple_health.score * 100).toFixed(1)}%)`
                : "model not available (ask Shira to train)"}
            </div>
          )}

          <div>
            <span className="font-semibold">YOLO boxes for:</span>{" "}
            {result.display_boxes_for}{" "}
            <span className="text-gray-500">({result.yolo.length})</span>
          </div>
        </div>
      )}
    </div>
  );
}
