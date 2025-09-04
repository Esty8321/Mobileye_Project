import { FlowerAnalysis } from '../types/flower';

// Mock flower types and analysis data
const flowerTypes = [
  {
    type: 'Rose',
    commonName: 'Garden Rose',
    scientificName: 'Rosa gallica',
    healthyRecommendations: [
      'Water deeply 2-3 times per week at the base of the plant',
      'Provide 6+ hours of morning sunlight daily',
      'Apply balanced fertilizer every 4-6 weeks during growing season',
      'Prune dead blooms regularly to encourage new growth'
    ],
    sickRecommendations: [
      'Remove affected leaves immediately to prevent spread',
      'Improve air circulation around the plant',
      'Apply fungicide treatment as needed',
      'Reduce watering frequency and water at soil level only'
    ]
  },
  {
    type: 'Sunflower',
    commonName: 'Common Sunflower',
    scientificName: 'Helianthus annuus',
    healthyRecommendations: [
      'Water deeply once per week, more in hot weather',
      'Ensure full sun exposure (8+ hours daily)',
      'Apply high-nitrogen fertilizer during early growth',
      'Support tall varieties with stakes if needed'
    ],
    sickRecommendations: [
      'Check for pest damage and treat accordingly',
      'Ensure proper drainage to prevent root rot',
      'Remove diseased plant material immediately',
      'Apply appropriate fungicide for detected diseases'
    ]
  },
  {
    type: 'Tulip',
    commonName: 'Garden Tulip',
    scientificName: 'Tulipa gesneriana',
    healthyRecommendations: [
      'Water moderately during growing season, avoid overwatering',
      'Provide bright, indirect sunlight',
      'Allow foliage to die back naturally for bulb energy',
      'Plant in well-draining soil to prevent bulb rot'
    ],
    sickRecommendations: [
      'Improve soil drainage immediately',
      'Remove infected bulbs to prevent spread',
      'Reduce watering frequency significantly',
      'Apply appropriate treatment for detected fungal issues'
    ]
  }
];

const healthIssues = [
  'Fungal infection detected on leaf surfaces',
  'Nutrient deficiency (likely nitrogen or phosphorus)',
  'Overwatering stress symptoms visible',
  'Pest damage on leaves and stems',
  'Insufficient sunlight exposure',
  'Root rot indicators present',
  'Bacterial infection symptoms',
  'Drought stress markers'
];

export const analyzeFlower = async (imageFile: File): Promise<FlowerAnalysis> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  const randomFlower = flowerTypes[Math.floor(Math.random() * flowerTypes.length)];
  const healthStatus = Math.random() > 0.3 ? 'healthy' : Math.random() > 0.5 ? 'sick' : 'critical';
  const confidence = Math.floor(Math.random() * 20) + 80; // 80-99%
  
  const issues = healthStatus !== 'healthy' 
    ? healthIssues.slice(0, Math.floor(Math.random() * 3) + 1)
    : undefined;
  
  const recommendations = healthStatus === 'healthy' 
    ? randomFlower.healthyRecommendations
    : randomFlower.sickRecommendations;

  return {
    id: crypto.randomUUID(),
    imageUrl: URL.createObjectURL(imageFile),
    healthStatus,
    confidence,
    flowerType: randomFlower.type,
    commonName: randomFlower.commonName,
    scientificName: randomFlower.scientificName,
    issues,
    careRecommendations: recommendations,
    analyzedAt: new Date()
  };
};