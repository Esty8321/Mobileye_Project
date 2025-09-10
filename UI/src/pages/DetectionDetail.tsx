import { useLocation, useNavigate } from "react-router-dom";
import type { PipelineDetection } from "../api";

export default function DetectionDetail() {
  const nav = useNavigate();
  const location = useLocation() as any;
  const det: PipelineDetection | undefined = location.state?.detection;
  const imageUrl: string | undefined = location.state?.imageUrl;

  if (!det) {
    return (
      <div className="p-6">
        <p>אין נתונים להצגה</p>
        <button onClick={() => nav(-1)}>חזרה</button>
      </div>
    );
  }

  return (
    <div className="p-6 grid gap-6 md:grid-cols-2">
      <div>
        <h2 className="text-xl font-semibold">פרטי הזיהוי</h2>
        <pre className="p-3 bg-gray-100 rounded">
{JSON.stringify(det, null, 2)}
        </pre>
        <button onClick={() => nav(-1)}>⬅ חזרה</button>
      </div>
      {imageUrl && <img src={imageUrl} alt="chosen" className="rounded shadow" />}
    </div>
  );
}
