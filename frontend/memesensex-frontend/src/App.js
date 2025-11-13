import "./App.css";
import { useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
//import { Client } from "@gradio/client";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import HowToUseSection from "./components/HowToUseSection";
import ToolSection from "./components/ToolSection";
import AboutSection from "./components/AboutSection";
import Footer from "./components/Footer";

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
      element.scrollIntoView({ behavior: "smooth" });
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
      { name: "Classification", duration: 1000 },
    ];

    for (let i = 0; i < stages.length; i++) {
      setCurrentStage(i);
      await new Promise((resolve) => setTimeout(resolve, stages[i].duration));
    }

	//const errorMessage = "";

    
    try {
      //Connect to the Flask client instead
      const formData = new FormData();
      formData.append("image", imageFile);
      
      const response = await fetch("/process_predict", {
        method: "POST",
        body: formData
      });

      // DEBUG LOGGING - Check response first
      console.log("Response status:", response.status);
      console.log("Response OK:", response.ok);

      // Parse JSON regardless of status (since your server returns JSON even on errors)
      let result;
      try {
        result = await response.json();
      } catch (jsonError) {
        console.error("Failed to parse JSON:", jsonError);
        throw new Error("Invalid response from server");
      }

      // DEBUG LOGGING      
      console.log("Full Result:", result);

      // Check for error in response FIRST - This is the key!
      if (result.error) {
        // Throw ONLY the error message
        throw new Error(result.error);
      }

      // Check if response was not OK (after checking for error message)
      if (!response.ok) {
        throw new Error(`Server error (${response.status})`);
      }

      // Check if data exists
      if (!result || !result.data) {
        console.error("Missing data in response. Full result:", result);
        throw new Error("Server returned incomplete data");
      }

      const resultData = result.data;

      // Check if prediction exists
      if (resultData.prediction === undefined || resultData.prediction === null) {
        console.error("Missing prediction in resultData:", resultData);
        throw new Error("Server did not return a prediction");
      }

      const predictionText = resultData.prediction;
      const isExplicit = predictionText.toLowerCase() === 'sexual' || 
                        predictionText.toLowerCase().includes('explicit');           

      const classificationResult = {
        classification: isExplicit ? "Explicit Content" : "Safe Content",
        details: {
          overall: isExplicit ? "explicit" : "safe",
          raw_text: resultData.raw_text || "N/A",
          clean_text: resultData.clean_text || "N/A",
          probabilities: resultData.probabilities ? [resultData.probabilities] : [[0, 0]],
        },
      };

      console.log("TRACER ROUND ======================================");
      console.log("Classification Result: ", classificationResult);

      setResults(classificationResult);
      
    } catch (error) {
      // This will now catch ALL errors including the TypeError
      console.error("=== ERROR CAUGHT ===");
      console.error("Error message:", error.message);
      
      // Display ONLY the error message (no "Error:" prefix if you don't want it)
      toast.error(error.message, {
        position: "top-center",
        autoClose: 4000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: false,
        progress: undefined,
        theme: "colored",
      });
      
      // Reset state
      setImage(null);
      setImageFile(null);
      setInputKey(Date.now());
      setResults(null);
      setIsLoading(false);
      setCurrentStage(0);
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
    if (files && files[0] && files[0].type.startsWith("image/")) {
      setImage(URL.createObjectURL(files[0]));
      setImageFile(files[0]);
    }
  };

  return (
    <div className="min-h-screen">
      <Navbar scrollToSection={scrollToSection} />
      <HeroSection scrollToSection={scrollToSection} />
      <HowToUseSection />
      <ToolSection
        image={image}
        imageFile={imageFile}
        inputKey={inputKey}
        isDragOver={isDragOver}
        isLoading={isLoading}
        currentStage={currentStage}
        results={results}
        handleImageChange={handleImageChange}
        handleClear={handleClear}
        handleClassify={handleClassify}
        handleDragOver={handleDragOver}
        handleDragLeave={handleDragLeave}
        handleDrop={handleDrop}
      />
      <AboutSection />
      <Footer />
      <ToastContainer />
    </div>
  );
}

export default App;
