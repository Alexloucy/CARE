import { Toaster } from 'sonner';
import { useColorMode } from '../features/theme/ThemeContext';

/**
 * Themed toaster component that responds to app theme (dark/light mode).
 * Uses glassmorphism styling for a modern, transparent look.
 */
const ThemedToaster = () => {
    const { colorTheme } = useColorMode();
    const isDark = colorTheme.mode === 'dark';

    return (
        <Toaster
            theme={isDark ? 'dark' : 'light'}
            position="bottom-right"
            style={{ zIndex: 100000 }}
            toastOptions={{
                style: {
                    background: isDark
                        ? 'rgba(30, 30, 30, 0.75)'
                        : 'rgba(255, 255, 255, 0.75)',
                    border: isDark
                        ? '1px solid rgba(255, 255, 255, 0.1)'
                        : '1px solid rgba(0, 0, 0, 0.1)',
                    color: isDark ? '#fff' : '#1a1a1a',
                    backdropFilter: 'blur(12px)',
                    WebkitBackdropFilter: 'blur(12px)',
                },
            }}
        />
    );
};

export default ThemedToaster;
