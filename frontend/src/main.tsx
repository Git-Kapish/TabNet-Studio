import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Apply persisted theme before paint to avoid light/dark flash.
const stored = localStorage.getItem('tabnet-studio-theme')
if (stored === 'dark' || stored === 'light') {
  document.documentElement.setAttribute('data-theme', stored)
} else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.setAttribute('data-theme', 'dark')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
