import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Self-hosted variable fonts (bundled by Vite; no network font loading).
// Fraunces with its optical-size axis for display type, Inter for body text.
import '@fontsource-variable/fraunces/opsz.css'
import '@fontsource-variable/inter'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
