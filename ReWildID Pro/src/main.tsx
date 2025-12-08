import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { CssBaseline } from '@mui/material'
import { ThemeContextProvider } from './features/theme/ThemeContext'
import { Provider } from 'react-redux'
import { store } from './store/initStore'
import { BrowserRouter } from 'react-router-dom'
import ThemedToaster from './components/ThemedToaster'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <Provider store={store}>
            <BrowserRouter>
                <ThemeContextProvider>
                    <CssBaseline />
                    <ThemedToaster />
                    <App />
                </ThemeContextProvider>
            </BrowserRouter>
        </Provider>
    </React.StrictMode>,
)
