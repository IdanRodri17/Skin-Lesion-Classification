import React, { useState, useCallback } from "react";
import "./App.css";

// Component for file upload with drag and drop
const ImageUpload = ({ onImageSelect, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onImageSelect(e.dataTransfer.files[0]);
      }
    },
    [onImageSelect]
  );

  const handleChange = useCallback(
    (e) => {
      e.preventDefault();
      if (e.target.files && e.target.files[0]) {
        onImageSelect(e.target.files[0]);
      }
    },
    [onImageSelect]
  );

  return (
    <div
      className={`upload-area ${dragActive ? "dragover" : ""}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => document.getElementById("fileInput").click()}
    >
      <div className="upload-icon">📷</div>
      <div className="upload-text">Drop your skin lesion image here</div>
      <div className="upload-subtext">
        or click to browse (JPG, PNG supported)
      </div>
      <button className="btn" disabled={isLoading}>
        {isLoading ? "Processing..." : "Choose Image"}
      </button>
      <input
        type="file"
        id="fileInput"
        accept="image/*"
        onChange={handleChange}
        style={{ display: "none" }}
      />
    </div>
  );
};

// Component for displaying prediction results
const Results = ({ results }) => {
  if (!results) return null;

  const { primary_prediction, predictions, severity } = results;

  return (
    <div className="results-section">
      {/* Primary Diagnosis */}
      <div className="prediction-card">
        <div className="prediction-title">🎯 Primary Diagnosis</div>
        <div className="prediction-result">
          <strong>{primary_prediction.class.toUpperCase()}</strong>
          <span
            className={`severity-indicator severity-${severity.medical_level}`}
          >
            Severity {severity.medical_level}/5
          </span>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${primary_prediction.confidence * 100}%` }}
          ></div>
        </div>
        <div style={{ textAlign: "center", marginTop: "10px" }}>
          <small>
            Confidence: {(primary_prediction.confidence * 100).toFixed(1)}%
          </small>
        </div>
      </div>

      {/* All Predictions */}
      <div className="prediction-card">
        <div className="prediction-title">📊 All Predictions</div>
        <div className="top-predictions">
          {predictions.map((pred, index) => (
            <div key={index} className="prediction-item">
              <span className="prediction-name">{pred.class}</span>
              <span className="prediction-confidence">
                {(pred.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Severity Assessment */}
      <div className="prediction-card">
        <div className="prediction-title">⚠️ Severity Assessment</div>
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "15px",
            }}
          >
            <span
              className={`severity-indicator severity-${severity.medical_level}`}
            >
              Level {severity.medical_level}/5
            </span>
            <span style={{ marginLeft: "15px", fontSize: "1.1em" }}>
              {severity.description}
            </span>
          </div>
          <p style={{ color: "#7f8c8d", fontSize: "0.9em", marginTop: "10px" }}>
            {severity.advice}
          </p>
        </div>
      </div>

      {/* Medical Disclaimer */}
      <div className="disclaimer">
        <h3>⚠️ Important Medical Disclaimer</h3>
        <p>
          <strong>
            This tool is for educational and research purposes only.
          </strong>
          It should not be used as a substitute for professional medical advice,
          diagnosis, or treatment. Always consult with a qualified healthcare
          provider for any medical concerns. The AI model may not be 100%
          accurate and should not be relied upon for medical decisions.
        </p>
      </div>
    </div>
  );
};

// Component for loading spinner
const LoadingSpinner = () => (
  <div className="loading">
    <div className="spinner"></div>
    <p>Analyzing image with AI model...</p>
  </div>
);

// Component for error messages
const ErrorMessage = ({ error, onClose }) => {
  if (!error) return null;

  return (
    <div className="error-message">
      {error}
      <button
        onClick={onClose}
        style={{
          marginLeft: "10px",
          background: "none",
          border: "none",
          color: "inherit",
          cursor: "pointer",
        }}
      >
        ✕
      </button>
    </div>
  );
};

// Main App Component
function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = useCallback((file) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setSelectedImage(file);
    setResults(null);
    setError(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const analyzeImage = useCallback(async () => {
    if (!selectedImage) {
      setError("Please select an image first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || "Prediction failed. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection and try again.");
      console.error("Analysis error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedImage]);

  return (
    <div className="App">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>🔬 AI Skin Lesion Classifier</h1>
          <p>
            Advanced deep learning model for skin lesion analysis using HAM10000
            dataset
          </p>
        </div>

        <div className="main-content">
          {/* Upload Section */}
          <div className="upload-section">
            <ImageUpload
              onImageSelect={handleImageSelect}
              isLoading={isLoading}
            />
          </div>

          {/* Image Preview */}
          {imagePreview && (
            <div className="preview-section">
              <h3
                style={{
                  textAlign: "center",
                  marginBottom: "20px",
                  color: "#2c3e50",
                }}
              >
                Image Preview
              </h3>
              <img src={imagePreview} alt="Preview" className="image-preview" />
              <div style={{ textAlign: "center", marginTop: "20px" }}>
                <button
                  className="btn"
                  onClick={analyzeImage}
                  disabled={isLoading}
                >
                  {isLoading ? "🔄 Analyzing..." : "🔍 Analyze Image"}
                </button>
              </div>
            </div>
          )}

          {/* Loading Spinner */}
          {isLoading && <LoadingSpinner />}

          {/* Error Message */}
          <ErrorMessage error={error} onClose={() => setError(null)} />

          {/* Results */}
          <Results results={results} />
        </div>
      </div>
    </div>
  );
}

export default App;
