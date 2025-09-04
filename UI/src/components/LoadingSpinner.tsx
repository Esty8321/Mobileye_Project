import React from 'react';

export const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-emerald-500"></div>
        <div className="absolute inset-0 rounded-full border-4 border-emerald-200"></div>
      </div>
      <p className="mt-4 text-gray-600 font-medium">
        Analyzing your flower with AI...
      </p>
      <p className="text-sm text-gray-500 mt-1">
        This may take a few moments
      </p>
    </div>
  );
};