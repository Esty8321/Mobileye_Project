import React from 'react';
import { Heart, AlertTriangle, AlertCircle, Leaf, Droplets, Sun, Calendar } from 'lucide-react';
import { FlowerAnalysis } from '../types/flower';

interface AnalysisResultsProps {
  analysis: FlowerAnalysis;
  onNewAnalysis: () => void;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ analysis, onNewAnalysis }) => {
  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <Heart className="text-emerald-500" size={24} />;
      case 'sick':
        return <AlertTriangle className="text-amber-500" size={24} />;
      case 'critical':
        return <AlertCircle className="text-red-500" size={24} />;
      default:
        return <Heart className="text-gray-400" size={24} />;
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-emerald-50 border-emerald-200 text-emerald-800';
      case 'sick':
        return 'bg-amber-50 border-amber-200 text-amber-800';
      case 'critical':
        return 'bg-red-50 border-red-200 text-red-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-emerald-600';
    if (confidence >= 70) return 'text-amber-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Image and Quick Status */}
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
        <div className="md:flex">
          <div className="md:w-1/2">
            <img
              src={analysis.imageUrl}
              alt="Analyzed flower"
              className="w-full h-64 md:h-full object-cover"
            />
          </div>
          <div className="md:w-1/2 p-6 space-y-4">
            <div className="flex items-center gap-3">
              {getHealthIcon(analysis.healthStatus)}
              <div>
                <h2 className="text-2xl font-bold text-gray-800 capitalize">
                  {analysis.healthStatus}
                </h2>
                <p className="text-gray-600">
                  {analysis.confidence}% confidence
                </p>
              </div>
            </div>

            <div className={`p-4 rounded-lg border-2 ${getHealthColor(analysis.healthStatus)}`}>
              <h3 className="font-semibold mb-1">{analysis.commonName}</h3>
              <p className="text-sm italic">{analysis.scientificName}</p>
            </div>

            <div className="flex justify-between text-sm text-gray-500">
              <span>Analyzed</span>
              <span>{analysis.analyzedAt.toLocaleDateString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Analysis */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Health Details */}
        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Heart className="text-emerald-500" size={24} />
            Health Analysis
          </h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Overall Health</span>
              <span className={`font-semibold capitalize ${getConfidenceColor(analysis.confidence)}`}>
                {analysis.healthStatus}
              </span>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-500 ${
                  analysis.healthStatus === 'healthy' ? 'bg-emerald-500' :
                  analysis.healthStatus === 'sick' ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${analysis.confidence}%` }}
              ></div>
            </div>

            {analysis.issues && analysis.issues.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-700 mb-2">Detected Issues:</h4>
                <ul className="space-y-1">
                  {analysis.issues.map((issue, index) => (
                    <li key={index} className="text-sm text-red-600 flex items-start gap-2">
                      <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                      {issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Care Recommendations */}
        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Leaf className="text-green-500" size={24} />
            Care Recommendations
          </h3>
          
          <div className="space-y-3">
            {analysis.careRecommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                <div className="p-1 bg-green-100 rounded-full">
                  {index === 0 && <Droplets size={16} className="text-green-600" />}
                  {index === 1 && <Sun size={16} className="text-green-600" />}
                  {index === 2 && <Calendar size={16} className="text-green-600" />}
                  {index > 2 && <Leaf size={16} className="text-green-600" />}
                </div>
                <p className="text-sm text-green-800">{recommendation}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="text-center">
        <button
          onClick={onNewAnalysis}
          className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
        >
          Analyze Another Flower
        </button>
      </div>
    </div>
  );
};