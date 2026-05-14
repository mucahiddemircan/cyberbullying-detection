import { useState, useEffect } from 'react'
import axios from 'axios'
import { Shield, AlertCircle, Search, RefreshCw, BarChart3, Info } from 'lucide-react'
import './App.css'

interface PredictionResult {
  prediction: string
  prediction_index: number
  probabilities: Record<string, number>
  model_used: string
  vectorizer_used: string
}

/**
 * Main Application Component
 */
function App() {
  // --- State Management ---
  const [text, setText] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [vectorizers, setVectorizers] = useState<string[]>([])
  const [selectedVectorizer, setSelectedVectorizer] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const API_BASE_URL = 'http://localhost:8000/api'

  // --- Effects ---
  useEffect(() => {
    fetchModels()
  }, [])

  // --- API Actions ---
  
  /**
   * Fetches available models from the backend and initializes selection state.
   */
  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`)
      const allModels = response.data as string[]
      setModels(allModels)
      
      // Extract unique vectorizers (BoW, TFIDF)
      const uniqueVectorizers = Array.from(new Set(allModels.map(m => m.split('_').pop() || '')))
      setVectorizers(uniqueVectorizers)
      
      if (uniqueVectorizers.length > 0) {
        setSelectedVectorizer(uniqueVectorizers[0])
        const filtered = allModels.filter(m => m.endsWith(uniqueVectorizers[0]))
        if (filtered.length > 0) {
          setSelectedModel(filtered[0])
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
      setError('Could not connect to the backend server.')
    }
  }

  /**
   * Handles vectorizer selection changes and synchronizes the selected model.
   */
  const handleVectorizerChange = (vectorizer: string) => {
    const currentBaseName = selectedModel.replace(`_${selectedVectorizer}`, '');
    setSelectedVectorizer(vectorizer);
    
    const filtered = models.filter(m => m.endsWith(vectorizer));
    if (filtered.length > 0) {
      const match = filtered.find(m => m.startsWith(currentBaseName));
      setSelectedModel(match || filtered[0]);
    }
  }

  /**
   * Submits the text for analysis using the selected model.
   */
  const handlePredict = async () => {
    if (!text.trim() || !selectedModel) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        text: text,
        model_name: selectedModel
      })
      setResult(response.data)
    } catch (err: any) {
      console.error('Prediction failed:', err)
      setError(err.response?.data?.detail || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  /**
   * Resets the input and result state.
   */
  const reset = () => {
    setText('')
    setResult(null)
    setError(null)
  }

  /**
   * Formats raw labels/ids into human-readable strings.
   */
  const formatLabel = (label: string) => {
    const mapping: Record<string, string> = {
      'ANN': 'Neural Network',
      'SVM': 'Support Vector Machine',
      'BoW': 'Bag of Words',
      'TFIDF': 'TF-IDF'
    }

    return mapping[label] || label.replace(/_/g, ' ')
  }

  return (
    <div className="app-wrapper">
      <header className="header">
        <h1>Cyberbullying Detection</h1>
        <p>Advanced AI-powered analysis for toxic social media content</p>
      </header>

      <div className="main-grid">
        {/* Input Section */}
        <main className="card">
          <div className="form-group">
            <label htmlFor="tweet-text">Tweet Content</label>
            <textarea
              id="tweet-text"
              placeholder="Paste the text you want to analyze here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              spellCheck="false"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="form-group">
              <label htmlFor="vectorizer-select">Embedding Method</label>
              <select
                id="vectorizer-select"
                value={selectedVectorizer}
                onChange={(e) => handleVectorizerChange(e.target.value)}
              >
                {vectorizers.map((v) => (
                  <option key={v} value={v}>
                    {formatLabel(v)}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="model-select">Analysis Model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models
                  .filter((m) => m.endsWith(selectedVectorizer))
                  .map((model) => {
                    const modelBaseName = model.replace(`_${selectedVectorizer}`, '');
                    return (
                      <option key={model} value={model}>
                        {formatLabel(modelBaseName)}
                      </option>
                    )
                  })}
              </select>
            </div>
          </div>

          <div className="actions">
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={loading || !text.trim()}
            >
              {loading ? <RefreshCw className="animate-spin" size={20} /> : <Search size={20} />}
              {loading ? 'Analyzing...' : 'Analyze Text'}
            </button>
            
            <button
              className="btn-secondary"
              onClick={reset}
              title="Reset"
            >
              <RefreshCw size={20} />
            </button>
          </div>

          {error && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              borderRadius: '8px', 
              background: '#fef2f2', 
              color: '#991b1b',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.875rem',
              border: '1px solid #fee2e2'
            }}>
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </main>

        {/* Results Section */}
        <section className="card">
          {!result && !loading && (
            <div className="placeholder-result">
              <Info size={48} strokeWidth={1.5} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <h3>No Analysis Yet</h3>
              <p>Enter some text and click Analyze to see the results here.</p>
            </div>
          )}

          {loading && (
            <div className="placeholder-result">
              <RefreshCw size={48} className="animate-spin" style={{ marginBottom: '1rem', color: '#6366f1' }} />
              <h3>Processing...</h3>
              <p>Our AI is analyzing the content for potential cyberbullying.</p>
            </div>
          )}

          {result && !loading && (
            <div className="result-content fade-in">
              <div className="result-header">
                {result.prediction === 'not_cyberbullying' ? (
                  <Shield size={32} color="#166534" />
                ) : (
                  <AlertCircle size={32} color="#991b1b" />
                )}
                <div style={{ flex: 1 }}>
                  <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Analysis Result</h3>
                  <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-slate-600)' }}>
                    Using {formatLabel(result.model_used.replace(`_${result.vectorizer_used}`, ''))}
                  </p>
                </div>
                <span className={`result-badge ${result.prediction === 'not_cyberbullying' ? 'badge-safe' : 'badge-danger'}`}>
                  {formatLabel(result.prediction)}
                </span>
              </div>

              <div className="scores-section">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                  <BarChart3 size={20} color="var(--primary)" />
                  <span style={{ fontWeight: 600 }}>Confidence Scores</span>
                </div>

                <div className="scores-list">
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([label, prob]) => (
                      <div key={label} className="score-item">
                        <div className="score-info">
                          <span>{formatLabel(label)}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="progress-bg">
                          <div 
                            className="progress-fill" 
                            style={{ 
                              width: `${prob * 100}%`,
                              backgroundColor: label === result.prediction 
                                ? (label === 'not_cyberbullying' ? '#22c55e' : '#ef4444') 
                                : 'var(--primary)'
                            }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
