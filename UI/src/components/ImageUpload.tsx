
// import { useEffect, useMemo, useState } from "react";
// import { predictPipeline, annotateImage, type PipelineDetection } from "../api";
// import type { PipelineResponse } from "../types/pipeline";
// import { useNavigate } from "react-router-dom";

// export default function ImageUpload() {
//   const [file, setFile] = useState<File | null>(null);
//   const [previewUrl, setPreviewUrl] = useState<string>("");
//   const [result, setResult] = useState<PipelineResponse | null>(null);

//   const [predicting, setPredicting] = useState(false);
//   const [saving, setSaving] = useState(false);
//   const [labelForAnnotate, setLabelForAnnotate] = useState("");

//   const navigate = useNavigate();

//   function onFileChange(f: File | null) {
//     setFile(f);
//     setResult(null);
//     setPredicting(false);
//     setSaving(false);

//     if (!f) {
//       if (previewUrl) URL.revokeObjectURL(previewUrl);
//       setPreviewUrl("");
//       return;
//     }
//     const url = URL.createObjectURL(f);
//     setPreviewUrl(url);
//   }

//   useEffect(() => {
//     return () => {
//       if (previewUrl) URL.revokeObjectURL(previewUrl);
//     };
//   }, [previewUrl]);

//   async function onPredict() {
//     if (!file || predicting) return;
//     setPredicting(true);
//     setResult(null);
//     try {
//       const res = await predictPipeline(file);
//       setResult(res);
//       console.log("API response:", res);
//     } catch (e) {
//       console.error(e);
//       alert("Prediction failed. Check backend and Network tab.");
//     } finally {
//       setPredicting(false);
//     }
//   }

//   async function onAnnotate() {
//     if (!file || !labelForAnnotate.trim() || saving) return;
//     setSaving(true);
//     try {
//       const res = await annotateImage(file, labelForAnnotate.trim());
//       if (res.ok) {
//         alert("Annotation saved!");
//         setLabelForAnnotate("");
//       } else {
//         alert("Annotate failed: " + (res.error || "unknown error"));
//       }
//     } catch (e) {
//       console.error(e);
//       alert("Annotate request failed.");
//     } finally {
//       setSaving(false);
//     }
//   }

//   const frameClass = useMemo(() => {
//     if (!result) return "border-gray-300";
//     if (result.classification.label !== "apple" || !result.apple_health) {
//       return "border-gray-300";
//     }
//     return result.apple_health.label === "healthy"
//       ? "border-emerald-500"
//       : "border-red-500";
//   }, [result]);

//   function onClickDetection(d: PipelineDetection) {
//     navigate("/detail", { state: { detection: d, imageUrl: previewUrl } });
//   }

//   return (
//     <div className="max-w-3xl mx-auto p-6 space-y-4">
//       <h1 className="text-2xl font-bold">Image Pipeline Demo</h1>

//       <div className="flex items-center gap-3">
//         <label className="px-3 py-2 bg-emerald-600 text-white rounded cursor-pointer">
//           Choose file
//           <input
//             type="file"
//             accept="image/*"
//             className="hidden"
//             onChange={(e) => onFileChange(e.target.files?.[0] || null)}
//           />
//         </label>
//         <button
//           onClick={onPredict}
//           disabled={!file || predicting}
//           className={`px-4 py-2 rounded ${
//             !file || predicting
//               ? "bg-gray-300 cursor-not-allowed text-gray-600"
//               : "bg-blue-600 text-white hover:bg-blue-700"
//           }`}
//         >
//           {predicting ? "Predicting..." : "Predict"}
//         </button>
//       </div>

//       {previewUrl && (
//         <div className="mt-3">
//           <div className={`relative inline-block rounded-lg border-4 ${frameClass}`}>
//             <img
//               src={previewUrl}
//               alt="preview"
//               className="block max-w-full h-auto rounded-lg"
//               style={{ maxHeight: 480, objectFit: "contain" }}
//             />

//             {/* YOLO boxes clickable */}
//             {result?.yolo?.map((d, i) => {
//               const [x1, y1, x2, y2] = d.box_norm; // XYXY normalized
//               return (
//                 <button
//                   key={i}
//                   title={`${d.label} ${(d.score * 100).toFixed(1)}%`}
//                   onClick={() => onClickDetection(d)}
//                   className="absolute"
//                   style={{
//                     left: `${x1 * 100}%`,
//                     top: `${y1 * 100}%`,
//                     width: `${(x2 - x1) * 100}%`,
//                     height: `${(y2 - y1) * 100}%`,
//                     border: "2px solid #0af",
//                     boxShadow: "0 0 0 1px rgba(0,0,0,0.6) inset",
//                   }}
//                 >
//                   <span
//                     className="absolute -top-5 left-0 text-[11px] bg-black text-white px-1.5 py-0.5 rounded"
//                     style={{ pointerEvents: "none" }}
//                   >
//                     {d.label} {(d.score * 100).toFixed(1)}%
//                   </span>
//                 </button>
//               );
//             })}
//           </div>
//         </div>
//       )}

//       <div className="flex gap-2">
//         <input
//           value={labelForAnnotate}
//           onChange={(e) => setLabelForAnnotate(e.target.value)}
//           placeholder="Annotate label (optional)"
//           className="flex-1 border rounded px-3"
//         />
//         <button
//           onClick={onAnnotate}
//           disabled={!file || !labelForAnnotate.trim() || saving}
//           className={`px-4 py-2 rounded ${
//             !file || !labelForAnnotate.trim() || saving
//               ? "bg-gray-300 cursor-not-allowed text-gray-600"
//               : "bg-emerald-600 text-white hover:bg-emerald-700"
//           }`}
//         >
//           {saving ? "Saving..." : "Annotate"}
//         </button>
//       </div>
//     </div>
//   );
// }


import { useEffect, useMemo, useState } from "react";
import { predictPipeline, annotateImage } from "../api";
import type { PipelineResponse, PipelineDetection } from "../types/pipeline";

import { useNavigate } from "react-router-dom";

export default function ImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PipelineResponse | null>(null);

  const [predicting, setPredicting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [labelForAnnotate, setLabelForAnnotate] = useState("");

  const navigate = useNavigate();

  function onFileChange(f: File | null) {
    setFile(f);
    setResult(null);
    setPredicting(false);
    setSaving(false);

    if (!f) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  }

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
      console.log("API response:", res);
    } catch (e) {
      console.error(e);
      alert("Prediction failed. Check backend and Network tab.");
    } finally {
      setPredicting(false);
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

  // (kept) outer frame class – uses whole-image health if present
  const frameClass = useMemo(() => {
    if (!result) return "border-gray-300";
    if (result.classification.label !== "apple" || !result.apple_health) {
      return "border-gray-300";
    }
    return result.apple_health.label === "healthy"
      ? "border-emerald-500"
      : "border-red-500";
  }, [result]);

  function onClickDetection(d: PipelineDetection) {
    navigate("/detail", { state: { detection: d, imageUrl: previewUrl } });
  }

  // NEW: decide per-box border color by apple_health
  function boxBorderColor(d: PipelineDetection) {
    // normalize to lower-case; treat missing apple_health as not healthy
    const lbl = d?.apple_health?.label?.toLowerCase() ?? "";
    return lbl.includes("healthy") ? "#22c55e" /* green-500 */ : "#ef4444" /* red-500 */;
  }

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
          <div className={`relative inline-block rounded-lg border-4 ${frameClass}`}>
            <img
              src={previewUrl}
              alt="preview"
              className="block max-w-full h-auto rounded-lg"
              style={{ maxHeight: 480, objectFit: "contain" }}
            />

            {/* YOLO boxes clickable, colored by per-box apple_health */}
            {result?.yolo?.map((d: PipelineDetection, i: number) => {
              const [x1, y1, x2, y2] = d.box_norm; // XYXY normalized
              const borderColor = boxBorderColor(d);
              const title = d.apple_health
                ? `${d.label} ${(d.score * 100).toFixed(1)}% • ${d.apple_health.label} (${(d.apple_health.score * 100).toFixed(1)}%)`
                : `${d.label} ${(d.score * 100).toFixed(1)}%`;

              return (
                <button
                  key={i}
                  title={title}
                  onClick={() => onClickDetection(d)}
                  className="absolute"
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                    border: `3px solid ${borderColor}`, // <-- colored per box
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.6) inset",
                    background: "transparent",
                    padding: 0,
                    cursor: "pointer",
                  }}
                >
                  {/* tiny floating label (non-interactive) */}
                  <span
                    className="absolute -top-5 left-0 text-[11px] bg-black text-white px-1.5 py-0.5 rounded"
                    style={{ pointerEvents: "none" }}
                  >
                    {d.apple_health?.label ?? d.label}
                  </span>
                </button>
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
    </div>
  );
}
