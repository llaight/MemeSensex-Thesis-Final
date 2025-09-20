import './App.css';
import logo from './asset/logo.svg';
import {useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [inputKey, setInputKey] = useState(Date.now());
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [results, setResults] = useState(null);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
      setImageFile(e.target.files[0]); 
    }
  };

  const handleClear = () => {
    setImage(null);
    setImageFile(null);
    setInputKey(Date.now()); 
    setResults(null);
    setIsLoading(false);
    setCurrentStage(0);
  };

  const handleClassify = async () => {
     if (!imageFile) return;
        setIsLoading(true);
        setResults(null);
        setCurrentStage(0);

        const stages = [
          { name: "Visual Analysis", duration: 2000 },
          { name: "Text Processing", duration: 1500 },
          { name: "Classification", duration: 1000 }
        ];

        for (let i = 0; i < stages.length; i++) {
          setCurrentStage(i);
          await new Promise(resolve => setTimeout(resolve, stages[i].duration));
        }

        try {
            const formData = new FormData();
            formData.append("image", imageFile); // <-- use the file from state

            const response = await fetch("http://127.0.0.1:5001/process_predict", {
              method: "POST",
              body: formData
            });

            const result = await response.json();
            if (!response.ok) {
              throw new Error(result.error || "Prediction failed.");
            }

            setResults({
              classification: result.data.prediction === "sexual" ? "Explicit Content" : "Safe Content",
              details: {
                overall: result.data.prediction === "sexual" ? "explicit" : "safe",
                raw_text: result.data.raw_text,
                clean_text: result.data.clean_text,
                probabilities: result.data.probabilities
              }
            });

          } catch (error) {
            console.error(error);
            alert("Error: " + error.message);
          }

        setIsLoading(false);
        setCurrentStage(0);

    };


  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0] && files[0].type.startsWith('image/')) {
      setImage(URL.createObjectURL(files[0]));
      setImageFile(files[0]); 
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-teal-50">
      {/* Navigation Header */}
      <nav className="bg-gradient-to-r from-purple-600 via-blue-600 to-teal-500 text-white py-4 px-6 shadow-lg fixed w-full top-0 z-50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white/100 rounded-full flex items-center justify-center">
              <img src={logo} alt="Logo" className="h-6 w-6"/>
            </div>
            <h1 className="text-2xl font-bold text-white">
              MemeSenseX
            </h1>
          </div>
          
          {/* Navigation Tabs */}
          <div className="flex items-center gap-6">
            <button 
              onClick={() => scrollToSection('home')}
              className="px-4 py-2 rounded-lg hover:bg-white/20 transition-all duration-300 font-medium"
            >
              Home
            </button>
            <button 
              onClick={() => scrollToSection('tool')}
              className="px-4 py-2 rounded-lg hover:bg-white/20 transition-all duration-300 font-medium"
            >
              Tool
            </button>
            <button 
              onClick={() => scrollToSection('about')}
              className="px-4 py-2 rounded-lg hover:bg-white/20 transition-all duration-300 font-medium"
            >
              About
            </button>
          </div>
        </div>
      </nav>

      {/* Home Section */}
      <section id="home" className="pt-20 min-h-screen flex items-center">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column - Text Content */}
            <div className="space-y-8">
              <div className="space-y-6">
                <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 via-blue-600 to-teal-500 bg-clip-text text-transparent">
                  MemeSenseX
                </h1>
                <p className="text-lg md:text-xl text-gray-600 leading-relaxed">
                  MemeSenseX is an AI-driven system built to detect sexually suggestive content in Tagalog memes. By combining advanced image recognition (ResNet-18) with natural language processing (Tagalog-BERT), it analyzes both visuals and text to capture the full meaning behind memes.
                </p>
              </div>
              
              <button 
                onClick={() => scrollToSection('tool')}
                className="bg-gradient-to-r from-red-500 to-yellow-500 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:from-red-600 hover:to-yellow-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Go to Tool →
              </button>
            </div>

            {/* Right Column - Logo */}
            <div className="flex justify-center lg:justify-end">
              <img 
                src={logo} 
                alt="MemeSenseX logo" 
                className="w-64 h-64 md:w-80 md:h-80 object-contain"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Tool Section */}
      <section id="tool" className="min-h-screen py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-800 mb-4">AI Analysis Tool</h2>
            <p className="text-gray-600 text-lg">Upload your meme and discover its content classification</p>
          </div>
          
          {/* Main Analysis Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Left Column - Input Image */}
            <section className="bg-white rounded-2xl shadow-xl overflow-hidden" aria-label="Image Upload Section">
              <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-white/20 rounded-lg flex items-center justify-center">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"/>
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold">Input Image</h3>
                </div>
                <p className="text-white/80 text-xs mt-1">Upload your meme for AI analysis</p>
              </div>
              
              <div className="p-6">
                {/* Drag and Drop Area */}
                <form onSubmit={(e) => e.preventDefault()}>
                  <div 
                    className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                      isDragOver 
                        ? 'border-purple-400 bg-purple-50' 
                        : 'border-gray-300 bg-gray-50 hover:border-purple-300 hover:bg-purple-25'
                    }`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                  {image ? (
                    <img
                      src={image}
                      alt="Uploaded"
                      className="w-full h-64 object-contain rounded-lg"
                    />
                  ) : (
                    <div className="space-y-4">
                      <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                        <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-lg font-medium text-gray-800">Drag & drop your meme here</p>
                        <p className="text-sm text-gray-500 mt-1">or</p>
                      </div>
                      
                      {/* Choose File button inside the drag zone */}
                      <div className="mt-4">
                        <input
                          key={inputKey}
                          type="file"
                          accept="image/*"
                          onChange={handleImageChange}
                          className="hidden"
                          id="file-upload"
                        />
                        <label 
                          htmlFor="file-upload"
                          className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-xl font-medium cursor-pointer hover:from-purple-700 hover:to-blue-700 transition-all duration-300 inline-flex items-center gap-2"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"/>
                          </svg>
                          Choose File
                        </label>
                      </div>
                    </div>
                  )}
                </div>

                </form>

                {/* Action Buttons */}
                <div className="mt-6 flex gap-3">
                  <button 
                    onClick={handleClassify}
                    disabled={!image || isLoading}
                    className={`px-6 py-3 rounded-xl font-medium flex-1 flex items-center justify-center gap-2 transition-all duration-300 ${
                      !image || isLoading 
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                        : 'bg-gradient-to-r from-purple-600 to-teal-500 text-white hover:from-purple-700 hover:to-teal-600'
                    }`}
                  >
                    {isLoading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"/>
                        </svg>
                        Classify Meme
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleClear}
                    disabled={isLoading}
                    className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                      isLoading 
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    Clear
                  </button>
                </div>
              </div>
            </section>

            {/* Right Column - Analysis Results */}
            <section className="bg-white rounded-2xl shadow-xl overflow-hidden" aria-label="Analysis Results Section">
              <div className="bg-gradient-to-r from-teal-500 to-green-500 text-white p-4">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-white/20 rounded-lg flex items-center justify-center">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"/>
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold">Analysis Results</h3>
                </div>
                <p className="text-white/80 text-xs mt-1">AI classification outcomes</p>
              </div>
              
              <div className="p-6">
                {isLoading ? (
                  // Loading State
                  <div className="text-center space-y-6">
                    <div className="relative w-24 h-24 mx-auto">
                      {/* Animated rotating squares */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="w-4 h-4 bg-purple-500 rounded animate-pulse"></div>
                          <div className="w-4 h-4 bg-blue-500 rounded animate-pulse" style={{animationDelay: '0.2s'}}></div>
                          <div className="w-4 h-4 bg-teal-500 rounded animate-pulse" style={{animationDelay: '0.4s'}}></div>
                          <div className="w-4 h-4 bg-green-500 rounded animate-pulse" style={{animationDelay: '0.6s'}}></div>
                        </div>
                      </div>
                      
                      {/* Outer rotating border */}
                      <div className="absolute inset-0 border-4 border-transparent border-t-purple-600 rounded-full animate-spin"></div>
                    </div>
                    
                    <div>
                      <h4 className="text-xl font-semibold text-gray-800 mb-2">Analyzing Content...</h4>
                      <p className="text-gray-600">
                        Our AI is processing visual and textual elements
                      </p>
                    </div>

                    {/* Progress Bar */}
                    <div className="space-y-4">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-600 via-blue-600 to-teal-500 h-2 rounded-full transition-all duration-1000 ease-out"
                          style={{ width: `${((currentStage + 1) / 3) * 100}%` }}
                        ></div>
                      </div>
                      
                      {/* Stage Labels */}
                      <div className="flex justify-between text-sm">
                        <div className={`flex items-center gap-2 ${currentStage >= 0 ? 'text-purple-600' : 'text-gray-400'}`}>
                          <div className={`w-3 h-3 rounded-full ${currentStage >= 0 ? 'bg-purple-600' : 'bg-gray-300'}`}></div>
                          <span className="font-medium">Visual Analysis</span>
                        </div>
                        <div className={`flex items-center gap-2 ${currentStage >= 1 ? 'text-blue-600' : 'text-gray-400'}`}>
                          <div className={`w-3 h-3 rounded-full ${currentStage >= 1 ? 'bg-blue-600' : 'bg-gray-300'}`}></div>
                          <span className="font-medium">Text Processing</span>
                        </div>
                        <div className={`flex items-center gap-2 ${currentStage >= 2 ? 'text-teal-600' : 'text-gray-400'}`}>
                          <div className={`w-3 h-3 rounded-full ${currentStage >= 2 ? 'bg-teal-600' : 'bg-gray-300'}`}></div>
                          <span className="font-medium">Classification</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : results ? (
                  // Results State
                  <div className="text-center space-y-6">
                    <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto ${
                      results.details.overall === 'safe' ? 'bg-green-100' : 'bg-red-100'
                    }`}>
                      <svg className={`w-10 h-10 ${
                        results.details.overall === 'safe' ? 'text-green-600' : 'text-red-600'
                      }`} fill="currentColor" viewBox="0 0 20 20">
                        {results.details.overall === 'safe' ? (
                          // Shield with checkmark icon for safe content
                          <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"/>
                        ) : (
                          // Warning/Alert triangle icon for explicit content
                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"/>
                        )}
                      </svg>
                    </div>
                    
                    {/* Classification directly under icon */}
                    <div className={`inline-block px-6 py-3 rounded-2xl font-semibold text-white ${
                      results.details.overall === 'safe' 
                        ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                        : 'bg-gradient-to-r from-orange-400 to-pink-500'
                    }`}>
                      {results.classification}
                    </div>
                    
                    <div>
                      <p className="text-gray-600 mb-6">
                        {results.details.overall === 'safe' 
                          ? 'This meme is appropriate for general audiences' 
                          : 'This meme contains explicit or inappropriate content'
                        }
                      </p>
                      
                      {/* Confidence Score */}
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium text-gray-600">Confidence Level</span>
                          <span className={`text-lg font-bold ${
                            results.details.overall === 'safe' ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {Math.round((results.details.overall === 'safe' ? results.details.probabilities[0][0] : results.details.probabilities[0][1]) * 100)}%
                          </span>
                        </div>
                        
                        {/* Confidence Bar */}
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                              results.details.overall === 'safe' ? 'bg-green-500' : 'bg-red-500'
                            }`}
                            style={{ 
                              width: `${Math.round((results.details.overall === 'safe' ? results.details.probabilities[0][0] : results.details.probabilities[0][1]) * 100)}%` 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    {/* Privacy Warning */}
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-12">
                      <div className="flex items-start gap-2">
                        <svg className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"/>
                        </svg>
                        <p className="text-xs text-blue-800">
                          <strong>Privacy Notice:</strong> This system does not store, save, or retain any uploaded images or analysis data. All processing is done locally and securely.
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  // Default State
                  <div className="text-center space-y-6">
                    <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto">
                      <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 12l2 2 4-4" />
                      </svg>
                    </div>
                    
                    <div>
                      <h4 className="text-xl font-semibold text-gray-800 mb-2">Ready for Analysis</h4>
                      <p className="text-gray-600">
                        Upload a meme image to get started with AI-powered content classification.
                      </p>
                    </div>

                    {/* AI Models Information */}
                    <div className="space-y-4 pt-6">
                      <h5 className="text-sm font-medium text-gray-500 mb-3 text-center">AI Models Used</h5>
                      
                      <div className="flex items-start gap-3 text-left">
                        <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <div className="w-3 h-3 bg-purple-600 rounded-full"></div>
                        </div>
                        <div>
                          <span className="text-gray-700 font-medium block">ResNet18</span>
                          <span className="text-xs text-gray-500">Image Feature Extraction</span>
                        </div>
                      </div>
                      
                      <div className="flex items-start gap-3 text-left">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
                        </div>
                        <div>
                          <span className="text-gray-700 font-medium block">TagalogBERT</span>
                          <span className="text-xs text-gray-500">Tagalog Text Processing & Understanding</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* Progress Indicators */}
          <div className="mt-8 flex justify-center">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-purple-600 rounded-full"></div>
                <span className="text-sm text-gray-600">Upload</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
                <span className="text-sm text-gray-600">Analyze</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-teal-600 rounded-full"></div>
                <span className="text-sm text-gray-600">Results</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="min-h-screen py-20 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-800 mb-4">About MemeSenseX</h2>
            <p className="text-gray-600 text-lg max-w-3xl mx-auto">
              MemeSenseX is an AI-driven system built to detect sexually suggestive content in Tagalog memes. By combining advanced image recognition with natural language processing, it analyzes both visuals and text to capture the full meaning behind memes.
            </p>
          </div>

          {/* The Stack Behind MemeSenseX */}
          <div className="mb-16">
            <h3 className="text-3xl font-bold text-center text-gray-800 mb-8">The Stack Behind MemeSenseX</h3>
            <p className="text-center text-gray-600 mb-12">Compact, powerful, and tuned for Filipino internet culture. Here's how each piece contributes.</p>
            
            <div className="grid md:grid-cols-3 gap-8">
              {/* ResNet-18 */}
              <div className="bg-gradient-to-br from-red-500 to-red-600 text-white rounded-2xl p-6">
                <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center mb-4">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <h4 className="text-xl font-bold mb-3">ResNet-18</h4>
                <p className="text-white/90 mb-4">A lightweight residual CNN that learns deeper visual patterns without degradation—effective for detecting suggestive cues in images.</p>
                <ul className="space-y-1 text-sm text-white/80">
                  <li>• Residual connections</li>
                  <li>• Lightweight CNN</li>
                  <li>• Robust visual features</li>
                  <li>• Efficient inference</li>
                </ul>
              </div>

              {/* Tagalog BERT */}
              <div className="bg-gradient-to-br from-orange-500 to-yellow-500 text-white rounded-2xl p-6">
                <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center mb-4">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <h4 className="text-xl font-bold mb-3">Tagalog BERT</h4>
                <p className="text-white/90 mb-4">BERT trained on large-scale Tagalog data to understand slang, code-mixing, and local expressions common in Filipino memes.</p>
                <ul className="space-y-1 text-sm text-white/80">
                  <li>• Tagalog-tuned</li>
                  <li>• Contextual text</li>
                  <li>• Slang aware</li>
                  <li>• Code-mix ready</li>
                </ul>
              </div>

              {/* MemeSenseX */}
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 text-white rounded-2xl p-6">
                <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center mb-4">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <h4 className="text-xl font-bold mb-3">MemeSenseX</h4>
                <p className="text-white/90 mb-4">Fuses ResNet-18 visuals with Tagalog BERT text via cross-attention and a contrastive multi-task loss for higher accuracy.</p>
                <ul className="space-y-1 text-sm text-white/80">
                  <li>• Cross-attention</li>
                  <li>• Contrastive loss</li>
                  <li>• Multimodal fusion</li>
                  <li>• High accuracy</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Multimodal Intelligence */}
          <div className="bg-gradient-to-r from-purple-600 to-teal-500 text-white rounded-2xl p-8 text-center">
            <h3 className="text-2xl font-bold mb-4">Multimodal Intelligence</h3>
            <p className="text-white/90 text-lg">
              Vision + Language + Culture: A multimodal system that reads, images, understands, tagalog, and interprets context.
            </p>
          </div>

          {/* Footer */}
          <div className="mt-16 pt-8 border-t border-gray-200 text-center">
            <div className="flex items-center justify-center gap-2 mb-4">
              <img src={logo} alt="Logo" className="h-8 w-8"/>
              <span className="text-xl font-bold text-gray-800">MemeSenseX</span>
            </div>
            <p className="text-gray-600">Powered by AICAD - Advanced AI for Content Analysis and Detection</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;