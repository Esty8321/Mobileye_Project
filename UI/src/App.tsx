import React from "react";
import { Header } from "./components/Header";
import { Features } from "./components/Features";
import ImageUpload from "./components/ImageUpload";

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50">
      <Header />

      <main className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            Upload Your Image
          </h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            the system will do classification and if it is apple she will check if healthy at last she will do the YOLO
          </p>
        </div>

        <ImageUpload />

        <div className="mt-16">
          <Features />
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-gray-300">© 2025 My App.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
