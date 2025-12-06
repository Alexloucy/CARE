import React, { createContext, useState, useMemo, useContext } from 'react';
import { ThemeProvider as MUIThemeProvider } from '@mui/material/styles';
import { getTheme } from '../../theme';

// Color theme definitions
export interface ColorTheme {
    id: string;
    name: string;
    mode: 'light' | 'dark';
    gradient: string;
    previewGradient: string; // For the swatch in settings
}

export const COLOR_THEMES: ColorTheme[] = [
    {
        id: 'default-dark',
        name: 'Default Dark',
        mode: 'dark',
        gradient: 'none',
        previewGradient: 'linear-gradient(135deg, #1C1C1E 0%, #2C2C2E 100%)',
    },
    {
        id: 'default-light',
        name: 'Default Light',
        mode: 'light',
        gradient: 'none',
        previewGradient: 'linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%)',
    },
    {
        id: 'aurora-green',
        name: 'Aurora Green',
        mode: 'dark',
        gradient: `
            linear-gradient(135deg, 
                #1a1b26 0%, 
                #1e2a3a 15%,
                #1a3a35 30%,
                #0f3a2f 45%,
                #1a4035 55%,
                #152a30 70%,
                #1a2535 85%,
                #151a25 100%
            )
        `,
        previewGradient: 'linear-gradient(135deg, #1a1b26 0%, #1a3a35 50%, #151a25 100%)',
    },
];

interface ThemeContextType {
    toggleColorMode: () => void;
    mode: 'light' | 'dark';
    colorTheme: ColorTheme;
    setColorTheme: (themeId: string) => void;
}

const defaultTheme = COLOR_THEMES.find(t => t.id === 'default-dark')!;

const ThemeContext = createContext<ThemeContextType>({
    toggleColorMode: () => { },
    mode: 'dark',
    colorTheme: defaultTheme,
    setColorTheme: () => { },
});

export const useColorMode = () => useContext(ThemeContext);

export const ThemeContextProvider = ({ children }: { children: React.ReactNode }) => {
    // Initialize color theme from localStorage
    const [colorThemeId, setColorThemeId] = useState<string>(() => {
        const savedThemeId = localStorage.getItem('colorTheme');
        if (savedThemeId && COLOR_THEMES.find(t => t.id === savedThemeId)) {
            return savedThemeId;
        }
        return 'default-dark';
    });

    const colorTheme = useMemo(() => {
        return COLOR_THEMES.find(t => t.id === colorThemeId) || defaultTheme;
    }, [colorThemeId]);

    // Mode is derived from the color theme
    const mode = colorTheme.mode;

    const colorMode = useMemo(
        () => ({
            toggleColorMode: () => {
                // Toggle between default-dark and default-light
                const newThemeId = mode === 'light' ? 'default-dark' : 'default-light';
                setColorThemeId(newThemeId);
                localStorage.setItem('colorTheme', newThemeId);
            },
            setColorTheme: (themeId: string) => {
                const theme = COLOR_THEMES.find(t => t.id === themeId);
                if (theme) {
                    setColorThemeId(themeId);
                    localStorage.setItem('colorTheme', themeId);
                }
            },
            mode,
            colorTheme,
        }),
        [mode, colorTheme],
    );

    const theme = useMemo(() => getTheme(mode), [mode]);

    return (
        <ThemeContext.Provider value={colorMode}>
            <MUIThemeProvider theme={theme}>
                {children}
            </MUIThemeProvider>
        </ThemeContext.Provider>
    );
};
