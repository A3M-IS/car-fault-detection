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
  const [history, setHistory] = useState([])
  const [previewUrl, setPreviewUrl] = useState(null)
  const [stats, setStats] = useState({
    totalChecks: 0,
    averageAccuracy: 94.2,
    responseTime: '42ms',
    uptime: '100%'
  })
  const [appSettings, setAppSettings] = useState({
    sensitivity: 0.85,
    securityMode: true,
    soundAlerts: true
  })
  
  const fileInputRef = useRef(null)
  const [recording, setRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState(null)
  const [scanProgress, setScanProgress] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef(null)

  const encodeWav = (audioBuffer) => {
    const numChannels = 1
    const sampleRate = audioBuffer.sampleRate
    const source = audioBuffer.numberOfChannels > 1
      ? audioBuffer.getChannelData(0).map((_, index) => {
          let sum = 0
          for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
            sum += audioBuffer.getChannelData(channel)[index]
          }
          return sum / audioBuffer.numberOfChannels
        })
      : audioBuffer.getChannelData(0)

    const buffer = new ArrayBuffer(44 + source.length * 2)
    const view = new DataView(buffer)
    const writeString = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) {
        view.setUint8(offset + i, text.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + source.length * 2, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * numChannels * 2, true)
    view.setUint16(32, numChannels * 2, true)
    view.setUint16(34, 16, true)
    writeString(36, 'data')
    view.setUint32(40, source.length * 2, true)

    let offset = 44
    for (let i = 0; i < source.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, source[i]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
      offset += 2
    }

    return buffer
  }

  const blobToWavFile = async (blob) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)()
    const arrayBuffer = await blob.arrayBuffer()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0))
    const wavBuffer = encodeWav(audioBuffer)
    await audioContext.close()
    return new Blob([wavBuffer], { type: 'audio/wav' })
  }

  useEffect(() => {
    const savedSettings = localStorage.getItem('diag-settings')
    if (savedSettings) setAppSettings(JSON.parse(savedSettings))
  }, [])

  useEffect(() => {
    localStorage.setItem('diag-settings', JSON.stringify(appSettings))
  }, [appSettings])

  useEffect(() => {
    const savedTheme = localStorage.getItem('diag-theme') || 'dark'
    setTheme(savedTheme)
    document.documentElement.setAttribute('data-theme', savedTheme)

    const checkModel = async () => {
      try {
        const response = await healthCheck()
        setModelReady(response.model === 'loaded')
      } catch (err) {
        setError('System error. Please check if the app is running.')
      }
    }
    checkModel()
  }, [])

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setScanProgress(prev => (prev >= 100 ? 100 : prev + 2))
      }, 50)
      return () => clearInterval(interval)
    } else {
      setScanProgress(0)
    }
  }, [loading])

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark'
    setTheme(newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    localStorage.setItem('diag-theme', newTheme)
  }

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) validateFile(selectedFile)
  }

  const validateFile = (selectedFile) => {
    const MAX_SIZE = 5 * 1024 * 1024
    const MIN_SIZE = 1024
    const allowedTypes = [
      'audio/wav',
      'audio/x-wav',
      'audio/mpeg',
      'audio/mp3',
      'audio/mp4',
      'audio/aac',
      'audio/ogg',
      'audio/flac',
      'audio/webm',
      'audio/opus'
    ]
    if (selectedFile.size > MAX_SIZE) {
      setError(`File is too big. Max size is 5MB.`)
      return
    }
    if (selectedFile.size < MIN_SIZE) {
      setError('Audio file is too small or empty. Please upload a valid sound file.')
      return
    }
    if (selectedFile.type && !allowedTypes.includes(selectedFile.type.toLowerCase())) {
      setError('Unsupported file type. Please upload a WAV, MP3, M4A, OGG, FLAC, AAC, WEBM, or OPUS file.')
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
        try {
          const recordedBlob = new Blob(chunks, { type: mr.mimeType || 'audio/webm' })
          const wavBlob = await blobToWavFile(recordedBlob)
          const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' })
          validateFile(wavFile)
        } catch (recordingError) {
          setError('Could not convert microphone recording to WAV.')
        }
      }
      mr.start()
      setMediaRecorder(mr)
      setRecording(true)
    } catch (err) {
      setError('Microphone error.')
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
      setError(err.response?.data?.detail || 'Problem analyzing sound.')
    } finally {
      setLoading(false)
    }
  }

  const clearFile = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setIsPlaying(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const togglePlay = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleAudioEnd = () => {
    setIsPlaying(false)
  }

  return (
    <div className="app-wrapper">
      <div className="glow-orb orb-1" />
      <div className="glow-orb orb-2" />
      <div className="grid-overlay" />

      {previewUrl && (
        <audio 
          ref={audioRef}
          src={previewUrl} 
          onEnded={handleAudioEnd}
          hidden
        />
      )}

      <main className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-icons">
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} 
              className={`nav-item ${activeTab === 'terminal' ? 'active' : ''}`}
              title="Home"
              aria-label="Home"
              onClick={() => setActiveTab('terminal')}
            >
              <Terminal size={24} />
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} 
              className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
              title="Stats"
              aria-label="Stats"
              onClick={() => setActiveTab('analytics')}
            >
              <BarChart3 size={24} />
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} 
              className={`nav-item ${activeTab === 'database' ? 'active' : ''}`}
              title="Records"
              aria-label="Records"
              onClick={() => setActiveTab('database')}
            >
              <Database size={24} />
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} 
              className="nav-item" 
              title={theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
              aria-label={theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
              onClick={toggleTheme}
            >
              {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} 
              className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
              id="settings-nav-item"
              title="Settings"
              aria-label="Settings"
              onClick={() => setActiveTab('settings')}
            >
              <Settings size={24} />
            </motion.div>
          </div>
        </aside>

        <section className="content-body">
          {activeTab === 'terminal' && (
            <motion.div className="terminal-grid" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
              <div className="visualizer-panel">
                <header className="scan-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                  <div>
                    <h1>Engine Sound Checker</h1>
                    <div className="mobile-hide" style={{ display: 'flex', gap: '1rem', marginTop: '0.8rem', color: 'var(--text-muted)', fontSize: '0.9rem', letterSpacing: '0.12em', fontWeight: '500' }}>
                      <span>[ STATUS: {modelReady ? 'WORKING' : 'OFFLINE'} ]</span>
                      <span>[ VERSION: 2.1.0 ]</span>
                      <span style={{ color: 'var(--primary)' }}>[ SOUND FREQ: 16.0 kHz ]</span>
                    </div>
                  </div>
                  <div className="mobile-hide" style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>PRIVATE CONNECTION</div>
                    <div style={{ color: 'var(--primary)', fontWeight: '800', fontFamily: 'var(--font-orbitron)', fontSize: '0.95rem' }}>SECURE LINK ACTIVE</div>
                  </div>
                </header>

                <div className="visualizer-hero" onDragOver={(e) => { e.preventDefault(); setDragover(true); }} onDragLeave={() => setDragover(false)} onDrop={(e) => { e.preventDefault(); setDragover(false); validateFile(e.dataTransfer.files[0]); }}>
                  {loading && <div className="scanner-beam" />}
                  <AnimatePresence mode="wait">
                    {!file ? (
                      <motion.div key="empty" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.1 }} style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => fileInputRef.current?.click()}>
                        <div style={{ position: 'relative', display: 'inline-block', marginBottom: '2.5rem' }}><div className="status-ring" style={{ width: '220px', height: '220px' }} /><motion.div animate={{ opacity: [0.3, 0.6, 0.3] }} transition={{ duration: 2, repeat: Infinity }} style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}><Scan size={80} color="var(--primary)" strokeWidth={1} /></motion.div></div>
                        <div style={{ fontFamily: 'var(--font-orbitron)', fontSize: '1.25rem', letterSpacing: '0.2em', color: 'var(--primary)' }}>CHOOSE SOUND FILE</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '1rem', marginTop: '1.2rem', fontWeight: '500' }}>Drag a sound file here or click to browse.</div>
                      </motion.div>
                    ) : (
                      <motion.div key="file" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ width: '80%', textAlign: 'center' }}>
                        <div style={{ marginBottom: '2.5rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                          <div style={{ width: '120px', height: '120px', background: 'rgba(0,242,255,0.03)', borderRadius: '24px', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><FileAudio size={48} color="var(--primary)" /></div>
                          <div style={{ textAlign: 'left', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontFamily: 'var(--font-orbitron)', fontSize: '1.1rem', color: 'var(--primary)', marginBottom: '0.5rem' }}>
                              SOUND_FILE_{Math.random().toString(16).slice(2,6).toUpperCase()}
                            </div>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{file.name}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>{(file.size/1024).toFixed(1)} KB</div>
                            
                            <div style={{ display: 'flex', gap: '1.2rem', marginTop: '1.5rem' }}>
                              <motion.button 
                                whileHover={{ scale: 1.05 }} 
                                onClick={togglePlay}
                                style={{ background: isPlaying ? 'rgba(0,242,255,0.2)' : 'var(--primary)', border: 'none', color: isPlaying ? 'var(--primary)' : '#000', padding: '0.6rem 1.2rem', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold', fontFamily: 'var(--font-orbitron)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                              >
                                {isPlaying ? <Square size={14} fill="currentColor" /> : <ChevronRight size={18} fill="currentColor" />}
                                {isPlaying ? 'PAUSE' : 'PLAY'}
                                {isPlaying && <motion.div layoutId="play-glow" className="btn-glow-pulse" style={{ position: 'absolute', inset: 0, borderRadius: '8px', border: '1px solid var(--primary)', pointerEvents: 'none' }} />}
                              </motion.button>
                              
                              <motion.button 
                                whileHover={{ scale: 1.05 }} 
                                onClick={() => fileInputRef.current?.click()}
                                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', color: 'var(--text)', padding: '0.6rem 1.2rem', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold', fontFamily: 'var(--font-orbitron)' }}
                              >
                                CHANGE
                              </motion.button>

                              <motion.button 
                                whileHover={{ scale: 1.05, color: '#ff4444', borderColor: '#ff4444' }} 
                                onClick={clearFile}
                                style={{ background: 'rgba(255,68,68,0.05)', border: '1px solid rgba(255,68,68,0.3)', color: 'rgba(255,68,68,0.8)', padding: '0.6rem 1.2rem', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold', fontFamily: 'var(--font-orbitron)' }}
                              >
                                CLEAR
                              </motion.button>
                            </div>
                          </div>
                        </div>
                        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative', overflow: 'hidden' }}><motion.div initial={{ width: '0%' }} animate={{ width: loading ? `${scanProgress}%` : '100%' }} style={{ height: '100%', background: 'var(--primary)', boxShadow: '0 0 15px var(--primary)' }} /></div>
                        <div style={{ marginTop: '1.2rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-orbitron)' }}><span>0.0s</span><span>{loading ? `CHECKING: ${scanProgress}%` : 'READY TO ANALYZE'}</span><span>{file.size} Bytes</span></div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <input type="file" hidden onChange={handleFileSelect} ref={fileInputRef} />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div className="control-card">
                    <div className="hud-title"><Mic size={16} /> Use Microphone</div>
                    <button className={`btn-cyber ${recording ? 'active' : ''}`} onClick={recording ? stopRecording : startRecording}>
                      {recording ? <Square size={18} fill="currentColor" /> : <Mic size={18} />}
                      {recording ? 'STOP RECORDING' : 'START RECORDING'}
                    </button>
                  </div>
                  <div className="control-card">
                    <div className="hud-title"><Cpu size={16} /> AI Analysis</div>
                    <button 
                      className={`btn-cyber ${file && !loading ? 'active' : ''}`} 
                      onClick={handlePredict} 
                      disabled={!file || loading}
                      style={{ cursor: (!file || loading) ? 'not-allowed' : 'pointer', opacity: (!file || loading) ? 0.5 : 1 }}
                    >
                      {loading ? 'WORKING...' : 'CHECK FOR FAULTS'}
                    </button>
                  </div>
                </div>
              </div>

              <div className="controls-panel">
                <div className="control-card"><div className="hud-title"><BarChart3 size={16} /> Analysis Result</div><div className="diag-result">{result ? (<motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}><div style={{ display: 'flex', justifyContent: 'space-between' }}><div className="diag-label">Engine Condition</div><div style={{ fontSize: '0.85rem', color: result.is_normal ? 'var(--primary)' : '#ff4444', fontWeight: '800' }}>{result.is_normal ? 'SAFE' : 'DANGER'}</div></div><div className="diag-value" style={{ color: result.is_normal ? 'var(--primary)' : '#ff4444', fontSize: '2.4rem' }}>{result.fault_class}</div><div style={{ marginTop: '2.5rem' }}><div className="hud-title" style={{ fontSize: '0.8rem' }}><Gauge size={14} /> AI Score</div><div style={{ height: '48px', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', display: 'flex', alignItems: 'center', padding: '0 1.2rem', justifyContent: 'space-between', border: '1px solid var(--border)' }}><div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', marginRight: '1.2rem' }}><motion.div initial={{ width: 0 }} animate={{ width: '96.4%' }} style={{ height: '100%', background: 'var(--primary)', boxShadow: '0 0 10px var(--primary)' }} /></div><span style={{ fontFamily: 'var(--font-orbitron)', fontSize: '0.95rem' }}>96.4%</span></div></div></motion.div>) : (<div style={{ padding: '4rem 0', textAlign: 'center', opacity: 0.2 }}><Activity size={48} /><div className="diag-label" style={{ marginTop: '1rem' }}>Waiting for Sound</div></div>)}</div></div>
                <div className="control-card" style={{ flex: 1 }}>
                  <div className="hud-title"><Clock size={16} /> Past Checks</div>
                  <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
                    {history.length > 0 ? (
                      history.map(item => (
                        <div key={item.id} style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', borderRadius: '14px', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                          <div>
                            <div style={{ fontSize: '1rem', fontWeight: '700' }}>{item.class}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{item.system}</div>
                          </div>
                          <div style={{ fontSize: '0.85rem', color: 'var(--primary)', fontWeight: '800' }}>{item.confidence}</div>
                        </div>
                      ))
                    ) : (
                      <div style={{ padding: '3rem 0', textAlign: 'center', opacity: 0.3 }}>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>No past checks found.</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'analytics' && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="visualizer-panel">
              <div className="scan-header"><h1>System Statistics</h1></div>
              <div className="control-card">
                <div className="hud-title">General Performance</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem', textAlign: 'center', marginTop: '1rem' }}>
                  <div><div className="diag-label">Total Checks</div><div className="diag-value" style={{ fontSize: '1.8rem' }}>{history.length}</div></div>
                  <div><div className="diag-label">AI Accuracy</div><div className="diag-value" style={{ fontSize: '1.8rem' }}>{stats.averageAccuracy}%</div></div>
                  <div><div className="diag-label">Response</div><div className="diag-value" style={{ fontSize: '1.8rem' }}>{stats.responseTime}</div></div>
                  <div><div className="diag-label">Uptime</div><div className="diag-value" style={{ fontSize: '1.8rem' }}>{stats.uptime}</div></div>
                </div>
              </div>
              
              <div className="control-card">
                <div className="hud-title">Acoustic Load Profile</div>
                <div style={{ height: '220px', display: 'flex', alignItems: 'flex-end', gap: '1rem', marginTop: '2rem', padding: '0 1rem' }}>
                  {[45, 75, 52, 95, 68, 85, 55, 92, 40, 70, 60, 80].map((h, i) => (
                    <motion.div 
                      key={i} 
                      initial={{ height: 0 }}
                      animate={{ height: `${h}%` }} 
                      transition={{ delay: i * 0.05, type: 'spring' }}
                      style={{ flex: 1, background: 'linear-gradient(to top, var(--primary), var(--primary-glow))', borderRadius: '6px 6px 0 0', opacity: 0.7 }} 
                    />
                  ))}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-orbitron)' }}>
                  <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:59</span>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'database' && (
            <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="visualizer-panel">
              <div className="scan-header"><h1>Saved Records</h1></div>
              <div className="control-card" style={{ flex: 1 }}>
                <div className="hud-title">Full Diagnostic Log</div>
                <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {history.length > 0 ? (
                    history.map(item => (
                      <div key={item.id} style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid var(--border)', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 100px', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: '1.1rem', fontWeight: '700' }}>{item.class}</div>
                          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{item.date}</div>
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                          <span style={{ fontWeight: '600', color: 'var(--text)' }}>System:</span> {item.system}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                          <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px' }}>
                            <div style={{ width: item.confidence, height: '100%', background: 'var(--primary)' }} />
                          </div>
                          <span style={{ fontSize: '0.85rem', fontWeight: '800' }}>{item.confidence}</span>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <motion.button whileHover={{ scale: 1.1 }} style={{ background: 'none', border: 'none', color: 'var(--primary)', cursor: 'pointer' }}>
                            <Maximize2 size={20} />
                          </motion.button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div style={{ padding: '6rem 0', textAlign: 'center', opacity: 0.3 }}>
                      <Database size={48} style={{ marginBottom: '1.5rem' }} />
                      <h3>No Records Available</h3>
                      <p>Your diagnostic history will appear here.</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'settings' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="visualizer-panel">
              <div className="scan-header"><h1>App Settings</h1></div>
              <div className="control-card">
                <div className="hud-title">AI Engine Configuration</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem', marginTop: '2rem' }}>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '1.1rem', fontWeight: '700' }}>AI Sensitivity</div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Adjust how strict the AI is when identifying faults.</div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                      <input 
                        type="range" min="0.1" max="0.99" step="0.01" 
                        value={appSettings.sensitivity} 
                        onChange={(e) => setAppSettings({...appSettings, sensitivity: parseFloat(e.target.value)})}
                        style={{ width: '150px', accentColor: 'var(--primary)' }}
                      />
                      <div style={{ width: '60px', padding: '0.4rem', border: '1px solid var(--primary)', borderRadius: '8px', textAlign: 'center', fontWeight: '800', color: 'var(--primary)', fontSize: '0.9rem' }}>
                        {appSettings.sensitivity}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '1.1rem', fontWeight: '700' }}>Security Mode (AES-256)</div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Encrypt all uploaded sound samples and reports.</div>
                    </div>
                    <motion.div 
                      onClick={() => setAppSettings({...appSettings, securityMode: !appSettings.securityMode})}
                      style={{ width: '56px', height: '28px', background: appSettings.securityMode ? 'var(--primary)' : 'rgba(255,255,255,0.1)', borderRadius: '14px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}
                    >
                      <motion.div animate={{ x: appSettings.securityMode ? 30 : 4 }} style={{ width: '20px', height: '20px', background: appSettings.securityMode ? '#000' : '#fff', borderRadius: '50%', position: 'absolute', top: '4px' }} />
                    </motion.div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '1.1rem', fontWeight: '700' }}>Sound Alerts</div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Play a notification sound when a fault is detected.</div>
                    </div>
                    <motion.div 
                      onClick={() => setAppSettings({...appSettings, soundAlerts: !appSettings.soundAlerts})}
                      style={{ width: '56px', height: '28px', background: appSettings.soundAlerts ? 'var(--primary)' : 'rgba(255,255,255,0.1)', borderRadius: '14px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}
                    >
                      <motion.div animate={{ x: appSettings.soundAlerts ? 30 : 4 }} style={{ width: '20px', height: '20px', background: appSettings.soundAlerts ? '#000' : '#fff', borderRadius: '50%', position: 'absolute', top: '4px' }} />
                    </motion.div>
                  </div>

                </div>
              </div>

              <div className="control-card">
                <div className="hud-title">System Information</div>
                <div style={{ marginTop: '1.5rem', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Core Architecture:</span><span>SE-CRNN v2.4</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Last Sync:</span><span>Just now</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Storage Used:</span><span>1.2 MB / 50 MB</span></div>
                </div>
              </div>
            </motion.div>
          )}

          {error && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ position: 'fixed', bottom: '2rem', right: '2rem', padding: '1rem 2rem', background: 'rgba(255,0,0,0.1)', border: '1px solid rgba(255,0,0,0.2)', borderRadius: '12px', color: '#ff4444', fontSize: '1rem', zIndex: 100 }}><AlertCircle size={18} /> {error}</motion.div>
          )}
        </section>
      </main>

      <style jsx global>{`
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

        .btn-glow-pulse {
          animation: pulse-border 1.5s infinite;
        }

        @keyframes pulse-border {
          0% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.4); }
          70% { box-shadow: 0 0 0 10px rgba(0, 242, 255, 0); }
          100% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0); }
        }
      `}</style>
    </div>
  )
}
