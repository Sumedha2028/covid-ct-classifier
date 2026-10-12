
import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function handleFileChange(event) {
    const selected = event.target.files?.[0];

    setResult(null);
    setError("");

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    if (!selected) {
      setFile(null);
      setPreview("");
      return;
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(selected.type)) {
      setFile(null);
      setPreview("");
      setError("Please select a JPEG, PNG, or WebP image.");

      if (event.target.value !== undefined) {
        event.target.value = "";
      }
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setFile(null);
      setPreview("");
      setError("Image size must be 10 MB or smaller.");

      if (event.target.value !== undefined) {
        event.target.value = "";
      }
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);

    const selected = event.dataTransfer.files?.[0];
    if (!selected) return;

    handleFileChange({
      target: {
        files: [selected],
        value: "",
      },
    });

    const input = document.getElementById("ct-file");
    if (input) input.value = "";
  }

  async function handlePredict() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to the API. Check that the backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setDragging(false);

    const input = document.getElementById("ct-file");
    if (input) input.value = "";
  }

  return (
    <main className="app">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">CT</span>
          <span>CT Classifier</span>
        </a>
        <span className="status">
          <span className="status-dot" />
          Local model
        </span>
      </header>

      <section className="hero">
        <p className="eyebrow">AI-ASSISTED IMAGE RESEARCH</p>
        <h1>CT Image Classifier</h1>
        <p className="subtitle">
          Upload a CT image to explore the output of a trained
          MobileNetV3-Large research model.
        </p>
      </section>

      <section className="workspace">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <span className="step">01</span>
            <div>
              <h2>Upload image</h2>
              <p>Select or drag a CT scan image from your device.</p>
            </div>
          </div>

          <label
            className={`dropzone ${dragging ? "dragging" : ""}`}
            htmlFor="ct-file"
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
              setDragging(true);
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) {
                setDragging(false);
              }
            }}
            onDrop={handleDrop}
          >
            {preview ? (
              <img className="preview" src={preview} alt="Selected CT scan" />
            ) : (
              <>
                <span className="upload-icon">↑</span>
                <strong>Choose or drop a CT image</strong>
                <span>JPEG, PNG or WebP · Maximum 10 MB</span>
                <span className="browse-button">Browse files</span>
              </>
            )}
          </label>

          <input
            id="ct-file"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
            hidden
          />

          {file && <p className="file-name">{file.name}</p>}

          <div className="actions">
            <button
              className="primary-button"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? "Analyzing image..." : "Run classification"}
            </button>
            <button
              className="secondary-button"
              onClick={handleReset}
              disabled={loading}
            >
              Reset
            </button>
          </div>

          {error && <p className="error-message" role="alert">{error}</p>}
        </div>

        <div className="panel results-panel">
          <div className="panel-heading">
            <span className="step">02</span>
            <div>
              <h2>Model output</h2>
              <p>Classification results appear here.</p>
            </div>
          </div>

          {!result && !loading && (
            <div className="empty-state">
              <span className="result-icon">⌁</span>
              <h3>Ready when you are</h3>
              <p>Upload an image and run classification to view the output.</p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="spinner" />
              <h3>Processing image</h3>
              <p>The model is generating its output.</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <p className="result-label">PREDICTED CLASS</p>
              <h3 className="predicted-class">{result.predicted_class}</h3>

              <div className="scores">
                <h4>Model output scores</h4>
                {Object.entries(result.probabilities).map(([name, score]) => (
                  <div className="score-row" key={name}>
                    <div className="score-label">
                      <span>{name}</span>
                      <strong>{Number(score).toFixed(2)}%</strong>
                    </div>
                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, Number(score)))}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <p className="result-filename">File: {result.filename}</p>
            </div>
          )}

          <p className="disclaimer">
            Research demonstration only. This model is not a medical device.
            Its outputs must not be used for diagnosis or treatment decisions.
          </p>
        </div>
      </section>

      <footer>
        <span>MobileNetV3-Large · PyTorch</span>
        <span>Experimental research project</span>
      </footer>
    </main>
  );
}

export default App;
