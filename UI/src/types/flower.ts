export interface FlowerAnalysis {
  id: string;
  imageUrl: string;
  healthStatus: 'healthy' | 'sick' | 'critical';
  confidence: number;
  flowerType: string;
  commonName: string;
  scientificName: string;
  issues?: string[];
  careRecommendations: string[];
  analyzedAt: Date;
}

export interface UploadedImage {
  file: File;
  preview: string;
}