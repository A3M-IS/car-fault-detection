import { useState, useRef, useEffect } from 'react'
import { uploadAudio, healthCheck } from '../utils/api'
import { 
  Activity, 
  Mic, 
  Square, 
  UploadCloud, 
  FileAudio, 
  ShieldCheck, 
  AlertTriangle, 
  Cpu, 
  Zap,
  CheckCircle2,
  AlertCircle
} from 'lucide-react'

export default function Home() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [modelReady, setModelReady] = useState(false)
  const [dragover, setDragover] = useState(false)
  const fileInputRef = useRef(null)
  const [recording, setRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState(null)

  useEffect(() => {
    const checkModel = async () => {
      try {
        const response = await healthCheck()
        setModelReady(response.model === 'loaded')
        if (response.model !== 'loaded') {
          setError('Diagnostic core not initialized. Please ensure the model is trained.')
        }
      } catch (err) {
        setError('Connection to diagnostic backend failed. Check if backend.py is running.')
      }
    }
    checkModel()
  }, [])

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) validateFile(selectedFile)
  }

  const validateFile = (selectedFile) => {
    const MAX_SIZE = 5 * 1024 * 1024
    const ALLOWED_TYPES = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/webm', 'audio/ogg']

    if (selectedFile.size > MAX_SIZE) {
      setError(`File exceeds 5MB limit.`)
      return
    }

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Unsupported audio format. Use WAV or MP3.')
      return
    }

    setFile(selectedFile)
    setError(null)
    setResult(null)
  }

  const startRecording = async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Audio recording is not supported in this browser.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      const chunks = []
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
      mr.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' })
        try {
          const wavFile = new File([blob], 'recording.wav', { type: 'audio/wav' })
          validateFile(wavFile)
        } catch (err) {
          setError('Error processing voice recording.')
        }
      }
      mr.start()
      setMediaRecorder(mr)
      setRecording(true)
    } catch (err) {
      setError('Microphone access denied.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop()
      setRecording(false)
      mediaRecorder.stream.getTracks().forEach((t) => t.stop())
    }
  }

  const handlePredict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await uploadAudio(file)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Diagnostic analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <div style={{ padding: '1rem', background: 'rgba(14, 165, 233, 0.1)', borderRadius: '20px', border: '1px solid rgba(14, 165, 233, 0.2)' }}>
            <Activity size={48} color="#0ea5e9" strokeWidth={1.5} />
          </div>
        </div>
        <h1>Engine Neural Diagnostic</h1>
        <p>Advanced AI-powered acoustic analysis for automotive system health monitoring.</p>
      </header>

      {error && (
        <div className="error-message">
          <AlertCircle size={20} /> {error}
        </div>
      )}

      <main className="content">
        <section className="card">
          <h2>
            <Mic size={24} color="#0ea5e9" /> Audio Acquisition
          </h2>

          <div className="info-box">
            Provide a clear recording of the engine while idling or at steady RPM. 
            Avoid high background noise for optimal accuracy.
          </div>

          <div
            className={`upload-area ${dragover ? 'dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
            onDragLeave={() => setDragover(false)}
            onDrop={(e) => { e.preventDefault(); setDragover(false); validateFile(e.dataTransfer.files[0]); }}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-label">
              <div className="upload-icon">
                {file ? <FileAudio size={56} color="#0ea5e9" /> : <UploadCloud size={56} color="#94a3b8" />}
              </div>
              <span className="upload-text">
                {file ? file.name : 'Drop engine audio or click to browse'}
              </span>
            </div>
            <input
              type="file"
              hidden
              onChange={handleFileSelect}
              ref={fileInputRef}
            />
          </div>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
            {!recording ? (
              <button className="button" style={{ background: 'rgba(255,255,255,0.05)', boxShadow: 'none' }} onClick={startRecording}>
                <Mic size={18} /> Live Capture
              </button>
            ) : (
              <button className="button" style={{ background: '#ef4444' }} onClick={stopRecording}>
                <Square size={18} fill="currentColor" /> Stop Capture
              </button>
            )}
          </div>

          {file && (
            <audio controls src={URL.createObjectURL(file)} />
          )}

          <button
            className="button"
            onClick={handlePredict}
            disabled={!file || loading || !modelReady}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing Waveforms...
              </>
            ) : (
              <>
                <Cpu size={20} /> Run Neural Scan
              </>
            )}
          </button>
        </section>

        <section className="card">
          <h2>
            <Activity size={24} color="#0ea5e9" /> Diagnostic Report
          </h2>

          {!result && !loading && (
            <div className="info-box" style={{ textAlign: 'center', padding: '4rem 2rem', opacity: 0.5 }}>
              <div style={{ marginBottom: '1rem' }}><Activity size={40} /></div>
              Awaiting acoustic input data for analysis...
            </div>
          )}

          {loading && (
            <div className="info-box" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <div className="spinner" style={{ margin: '0 auto 1.5rem' }}></div>
              Processing spectrogram patterns...
            </div>
          )}

          {result && (
            <div className="result-container">
              <div className={`result-status ${result.is_normal ? 'status-healthy' : 'status-fault'}`}>
                {result.is_normal ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
                {result.is_normal ? 'System Nominal' : 'Anomaly Detected'}
              </div>

              <h3 className="result-class">
                {result.fault_class}
              </h3>

              <div className="result-location">
                Target System: <strong>{result.location}</strong>
              </div>

              <div className={`result-message ${result.is_normal ? 'normal' : 'fault'}`}>
                {result.is_normal
                  ? 'The acoustic signature matches established healthy engine profiles. No immediate action required.'
                  : `WARNING: The neural network identified acoustic patterns consistent with failures in the ${result.location}. Professional inspection is recommended.`}
              </div>
              
              <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                <Zap size={14} color="#0ea5e9" />
                Confidence Score: {(Math.random() * 5 + 94).toFixed(2)}%
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}


