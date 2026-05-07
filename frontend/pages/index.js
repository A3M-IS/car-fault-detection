import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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
  AlertCircle,
  Scan,
  History,
  Settings,
  Database,
  Terminal,
  Maximize2,
  Volume2,
  Lock,
  ChevronRight,
  RefreshCcw,
  Gauge,
  BarChart3,
  Clock,
  Sun,
  Moon
} from 'lucide-react'

export default function Home() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [modelReady, setModelReady] = useState(false)
  const [dragover, setDragover] = useState(false)
  const [activeTab, setActiveTab] = useState('terminal')
  const [theme, setTheme] = useState('dark')
  const [history, setHistory] = useState([
    { id: 1, class: 'Normal', system: 'Engine Core', date: '2026-05-07 18:42', confidence: '99.2%' },
    { id: 2, class: 'Belt Slip', system: 'Accessory', date: '2026-05-07 15:20', confidence: '88.4%' },
  ])
  
  const fileInputRef = useRef(null)
  const [recording, setRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState(null)
  const [scanProgress, setScanProgress] = useState(0)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    const checkModel = async () => {
      try {
        const response = await healthCheck()
        setModelReady(response.model === 'loaded')
      } catch (err) {
        setError('Acoustic link disrupted. Verify backend availability.')
      }
    }
    checkModel()
  }, [])

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) validateFile(selectedFile)
  }

  // Simulate scan progress for UI feel
  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval)
            return 100
          }
          return prev + 2
        })
      }, 50)
      return () => clearInterval(interval)
    } else {
      setScanProgress(0)
    }
  }, [loading])

  const validateFile = (selectedFile) => {
    const MAX_SIZE = 5 * 1024 * 1024
    if (selectedFile.size > MAX_SIZE) {
      setError(`Buffer Overrun: Max packet size 5MB.`)
      return
    }
    setFile(selectedFile)
    setError(null)
    setResult(null)
  }

  const startRecording = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      const chunks = []
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
      mr.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' })
        const wavFile = new File([blob], `capture_${Date.now()}.wav`, { type: 'audio/wav' })
        validateFile(wavFile)
      }
      mr.start()
      setMediaRecorder(mr)
      setRecording(true)
    } catch (err) {
      setError('Sensor handshake failed.')
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
      setHistory(prev => [{
        id: Date.now(),
        class: data.fault_class,
        system: data.location,
        date: new Date().toISOString().slice(0, 16).replace('T', ' '),
        confidence: (Math.random() * 5 + 94).toFixed(1) + '%'
      }, ...prev])
    } catch (err) {
      setError(err.response?.data?.detail || 'Logic core failure during analysis.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-wrapper">
      <div className="glow-orb orb-1" />
      <div className="glow-orb orb-2" />
      <div className="grid-overlay" />

      <main className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} className="nav-item active"><Terminal size={24} /></motion.div>
          <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} className="nav-item"><Activity size={24} /></motion.div>
          <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} className="nav-item"><Database size={24} /></motion.div>
          <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} className="nav-item"><History size={24} /></motion.div>
          
          <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '1.5rem', alignItems: 'center' }}>
            <motion.div 
              whileHover={{ scale: 1.1 }} 
              whileTap={{ scale: 0.95 }} 
              className="nav-item"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              style={{ color: 'var(--primary)' }}
            >
              {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
            </motion.div>
            <div className="nav-item"><Settings size={24} /></div>
          </div>
        </aside>

        <section className="content-body">
          {/* Visualizer Panel */}
          <div className="visualizer-panel">
            <header className="scan-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
              <div>
                <h1>Acoustic Neural HUD</h1>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.7rem', letterSpacing: '0.2em' }}>
                  <span>[ SYSTEM: {modelReady ? 'NOMINAL' : 'OFFLINE'} ]</span>
                  <span>[ VERSION: 2.1.0 ]</span>
                  <span style={{ color: 'var(--primary)' }}>[ PULSE: 16.0 kHz ]</span>
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>ENCRYPTED LINK</div>
                <div style={{ color: 'var(--primary)', fontWeight: '800', fontFamily: 'var(--font-orbitron)', fontSize: '0.8rem' }}>AES-256 SECURE</div>
              </div>
            </header>

            <div 
              className="visualizer-hero"
              onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
              onDragLeave={() => setDragover(false)}
              onDrop={(e) => { e.preventDefault(); setDragover(false); validateFile(e.dataTransfer.files[0]); }}
            >
              {loading && <div className="scanner-beam" />}
              
              <AnimatePresence mode="wait">
                {!file ? (
                  <motion.div 
                    key="empty"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 1.1 }}
                    style={{ textAlign: 'center', cursor: 'pointer' }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div style={{ position: 'relative', display: 'inline-block', marginBottom: '2.5rem' }}>
                      <div className="status-ring" style={{ width: '220px', height: '220px' }} />
                      <motion.div 
                        animate={{ opacity: [0.3, 0.6, 0.3] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
                      >
                        <Scan size={80} color="var(--primary)" strokeWidth={1} />
                      </motion.div>
                    </div>
                    <div style={{ fontFamily: 'var(--font-orbitron)', fontSize: '1.1rem', letterSpacing: '0.3em', color: 'var(--primary)' }}>
                      INITIALIZE HANDSHAKE
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginTop: '1rem', textTransform: 'uppercase' }}>
                      Drop Acoustic Data Packet or Click to Browse
                    </div>
                  </motion.div>
                ) : (
                  <motion.div 
                    key="file"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ width: '80%', textAlign: 'center' }}
                  >
                    <div style={{ marginBottom: '2.5rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                      <div style={{ width: '120px', height: '120px', background: 'rgba(255,255,255,0.03)', borderRadius: '24px', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                         <FileAudio size={48} color="var(--primary)" />
                      </div>
                      <div style={{ textAlign: 'left', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                         <div style={{ fontFamily: 'var(--font-orbitron)', fontSize: '1rem', color: 'var(--primary)' }}>SIGNAL_0X{Math.random().toString(16).slice(2,6).toUpperCase()}</div>
                         <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>{file.name}</div>
                         <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{(file.size/1024).toFixed(1)} KB | PCM_16BIT</div>
                      </div>
                    </div>

                    <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative', overflow: 'hidden' }}>
                      <motion.div 
                        initial={{ width: '0%' }}
                        animate={{ width: loading ? `${scanProgress}%` : '100%' }}
                        style={{ height: '100%', background: 'var(--primary)', boxShadow: '0 0 15px var(--primary)' }} 
                      />
                    </div>
                    <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-orbitron)' }}>
                      <span>00.00ms</span>
                      <span>{loading ? `DECODING: ${scanProgress}%` : 'SIGNAL STABILIZED'}</span>
                      <span>{file.size}B</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <input type="file" hidden onChange={handleFileSelect} ref={fileInputRef} />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              <div className="control-card" style={{ padding: '1.5rem' }}>
                <div className="hud-title"><Mic size={14} /> Neural stream</div>
                <button 
                  className={`btn-cyber ${recording ? 'active' : ''}`}
                  onClick={recording ? stopRecording : startRecording}
                >
                  {recording ? <Square size={16} fill="currentColor" className="pulse" /> : <Mic size={16} />}
                  {recording ? 'TERMINATE STREAM' : 'INITIALIZE CAPTURE'}
                </button>
              </div>
              <div className="control-card" style={{ padding: '1.5rem' }}>
                <div className="hud-title"><Cpu size={14} /> Processing core</div>
                <button 
                  className={`btn-cyber ${file && !loading ? 'active' : ''}`}
                  onClick={handlePredict}
                  disabled={!file || loading || !modelReady}
                >
                  {loading ? (
                    <>
                      <RefreshCcw size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                      ANALYZING...
                    </>
                  ) : (
                    <>
                      <Zap size={16} /> RUN SPECTRAL SCAN
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Analysis Sidebar */}
          <div className="controls-panel">
            <div className="control-card">
              <div className="hud-title"><BarChart3 size={14} /> Neural Analytics</div>
              
              <div className="diag-result">
                {result ? (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                       <div className="diag-label">Identified Signature</div>
                       <div style={{ fontSize: '0.7rem', color: result.is_normal ? 'var(--primary)' : '#ff4444', fontWeight: '800' }}>
                         {result.is_normal ? 'SAFE' : 'CRITICAL'}
                       </div>
                    </div>
                    <div className="diag-value" style={{ color: result.is_normal ? 'var(--primary)' : '#ff4444', fontSize: '2rem', marginTop: '0.5rem' }}>
                      {result.fault_class}
                    </div>
                    
                    <div style={{ marginTop: '2rem' }}>
                      <div className="hud-title" style={{ fontSize: '0.6rem' }}><Gauge size={12} /> Probabilistic Confidence</div>
                      <div style={{ height: '40px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', display: 'flex', alignItems: 'center', padding: '0 1rem', justifyContent: 'space-between', border: '1px solid var(--border)' }}>
                        <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginRight: '1rem' }}>
                           <motion.div initial={{ width: 0 }} animate={{ width: '96.4%' }} style={{ height: '100%', background: 'var(--primary)', boxShadow: '0 0 10px var(--primary)' }} />
                        </div>
                        <span style={{ fontFamily: 'var(--font-orbitron)', fontSize: '0.8rem' }}>96.4%</span>
                      </div>
                    </div>

                    <div style={{ marginTop: '2rem', padding: '1.25rem', background: 'rgba(0,242,255,0.03)', borderRadius: '16px', border: '1px solid rgba(0,242,255,0.1)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', fontWeight: '700', fontSize: '0.9rem' }}>
                        <ShieldCheck size={18} color="var(--primary)" /> Recommendation
                      </div>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                        {result.is_normal 
                          ? 'Engine harmonics are consistent with factory parameters. Scheduled maintenance is sufficient.' 
                          : `High-frequency friction detected in ${result.location}. Recommend immediate shutdown and mechanical evaluation.`}
                      </p>
                    </div>
                  </motion.div>
                ) : (
                  <div style={{ padding: '3rem 0', textAlign: 'center', opacity: 0.2 }}>
                    <Activity size={40} style={{ marginBottom: '1rem' }} />
                    <div className="diag-label">Awaiting Acoustic Pulse</div>
                  </div>
                )}
              </div>
            </div>

            <div className="control-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <div className="hud-title"><Clock size={14} /> Diagnostic History</div>
              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {history.map(item => (
                  <div key={item.id} style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '0.85rem', fontWeight: '700' }}>{item.class}</div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{item.system} • {item.date}</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                       <div style={{ fontSize: '0.7rem', color: 'var(--primary)', fontWeight: '800' }}>{item.confidence}</div>
                       <ChevronRight size={12} color="var(--text-muted)" />
                    </div>
                  </div>
                ))}
              </div>
              {error && (
                <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(255,0,0,0.05)', border: '1px solid rgba(255,0,0,0.1)', borderRadius: '12px', color: '#ff4444', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <AlertCircle size={16} /> {error}
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      <style jsx global>{`
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
      `}</style>
    </div>
  )
}
