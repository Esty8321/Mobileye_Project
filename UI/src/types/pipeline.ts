// UI/src/types/pipeline.ts
export interface AppleHealth {
  label: string;
  score: number;
}

export interface PipelineDetection {
  label: string;
  score: number;
  box: [number, number, number, number];
  box_norm: [number, number, number, number];
  apple_health?: AppleHealth; // <-- הוספנו את השדה האופציונלי
}

export interface Classification {
  label: string;
  score: number;
  index?: number;
}

export interface PipelineResponse {
  classification: Classification;
  apple_health?: AppleHealth; // אם את עדיין מחזירה ברמת תמונה שלמה
  yolo: PipelineDetection[];
  display_boxes_for: string;
  meta: {
    cls_thresh: number;
    yolo_conf: number;
    yolo_iou: number;
  };
}
