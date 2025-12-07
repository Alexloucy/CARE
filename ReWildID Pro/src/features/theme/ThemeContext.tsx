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
    special?: 'prismatic-burst' | 'color-bends' | 'floating-lines' | 'galaxy' | 'light-pillar'; // For animated WebGL backgrounds
    image?: string; // For static image backgrounds
}

export const COLOR_THEMES: ColorTheme[] = [
    // Default themes (no gradient)
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
    // Dark themes
    {
        id: 'aurora-green',
        name: 'Aurora',
        mode: 'dark',
        gradient: 'linear-gradient(135deg, #1a1b26 0%, #1e2a3a 15%, #1a3a35 30%, #0f3a2f 45%, #1a4035 55%, #152a30 70%, #1a2535 85%, #151a25 100%)',
        previewGradient: 'linear-gradient(135deg, #1a1b26 0%, #1a3a35 50%, #151a25 100%)',
    },
    {
        id: 'megatron',
        name: 'MegaTron',
        mode: 'dark',
        gradient: 'linear-gradient(to right, #2C5364, #203A43, #0F2027)',
        previewGradient: 'linear-gradient(to right, #2C5364, #203A43, #0F2027)',
    },
    {
        id: 'forest',
        name: 'Forest',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #2C7744, #5A3F37)',
        previewGradient: 'linear-gradient(to left, #2C7744, #5A3F37)',
    },
    {
        id: 'frost',
        name: 'Frost',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #004e92, #000428)',
        previewGradient: 'linear-gradient(to left, #004e92, #000428)',
    },
    {
        id: 'lawrencium',
        name: 'Lawrencium',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #24243e, #302b63, #0f0c29)',
        previewGradient: 'linear-gradient(to left, #24243e, #302b63, #0f0c29)',
    },
    {
        id: 'midnight-city',
        name: 'Midnight City',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #414345, #232526)',
        previewGradient: 'linear-gradient(to left, #414345, #232526)',
    },
    {
        id: 'royal',
        name: 'Royal',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #243B55, #141E30)',
        previewGradient: 'linear-gradient(to left, #243B55, #141E30)',
    },
    // Special animated themes
    {
        id: 'prismatic-burst',
        name: 'Prismatic',
        mode: 'dark',
        gradient: 'none', // Handled specially by Layout
        previewGradient: 'linear-gradient(135deg, #ff007a 0%, #4d3dff 50%, #00ffff 100%)',
        special: 'prismatic-burst',
    },
    {
        id: 'color-bends',
        name: 'Color Bends',
        mode: 'dark',
        gradient: 'none', // Handled specially by Layout
        previewGradient: 'linear-gradient(135deg, #ff5c7a 0%, #8a5cff 50%, #00ffd1 100%)',
        special: 'color-bends',
    },
    {
        id: 'floating-lines',
        name: 'Floating Lines',
        mode: 'dark',
        gradient: 'none', // Handled specially by Layout
        previewGradient: 'linear-gradient(135deg, #e947f5 0%, #2f4ba2 50%, #000000 100%)',
        special: 'floating-lines',
    },
    {
        id: 'galaxy',
        name: 'Galaxy',
        mode: 'dark',
        gradient: 'none', // Handled specially by Layout
        previewGradient: 'radial-gradient(circle, #1a1a2e 0%, #16213e 50%, #0f0f23 100%)',
        special: 'galaxy',
    },
    {
        id: 'light-pillar',
        name: 'Light Pillar',
        mode: 'dark',
        gradient: 'none', // Handled specially by Layout
        previewGradient: 'linear-gradient(180deg, #5227FF 0%, #FF9FFC 100%)',
        special: 'light-pillar',
    },
    // Image themes
    {
        id: 'minecraft',
        name: 'Minecraft',
        mode: 'dark',
        gradient: 'none',
        previewGradient: 'url(/images/mc.jpg)',
        image: '/images/mc.jpg',
    },
    // Light themes
    {
        id: 'moonlit-asteroid',
        name: 'Moonlit',
        mode: 'light',
        gradient: 'linear-gradient(to right, #78ffd6, #a8ff78)',
        previewGradient: 'linear-gradient(to right, #78ffd6, #a8ff78)',
    },
    {
        id: 'summer-dog',
        name: 'Summer',
        mode: 'light',
        gradient: 'linear-gradient(to right, #FAFFD1, #A1FFCE)',
        previewGradient: 'linear-gradient(to right, #FAFFD1, #A1FFCE)',
    },
    {
        id: 'limade',
        name: 'Limade',
        mode: 'light',
        gradient: 'linear-gradient(to left, #ED8F03, #FFB75E)',
        previewGradient: 'linear-gradient(to left, #ED8F03, #FFB75E)',
    },
    {
        id: 'cool-sky',
        name: 'Cool Sky',
        mode: 'light',
        gradient: 'linear-gradient(to right, #FFFFFF, #6DD5FA, #2980B9)',
        previewGradient: 'linear-gradient(to right, #FFFFFF, #6DD5FA, #2980B9)',
    },
    {
        id: 'margo',
        name: 'Margo',
        mode: 'light',
        gradient: 'linear-gradient(to left, #FFFFFF, #FFEFBA)',
        previewGradient: 'linear-gradient(to left, #FFFFFF, #FFEFBA)',
    },
    {
        id: 'dance-to-forget',
        name: 'Dance',
        mode: 'light',
        gradient: 'linear-gradient(to left, #F9D423, #FF4E50)',
        previewGradient: 'linear-gradient(to left, #F9D423, #FF4E50)',
    },
    {
        id: 'megatron-light',
        name: 'MegaTron',
        mode: 'light',
        gradient: 'linear-gradient(to right, #f7797d, #FBD786, #C6FFDD)',
        previewGradient: 'linear-gradient(to right, #f7797d, #FBD786, #C6FFDD)',
    },
    // More dark themes
    {
        id: 'moss',
        name: 'Moss',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #71B280, #134E5E)',
        previewGradient: 'linear-gradient(to left, #71B280, #134E5E)',
    },
    {
        id: 'meridan',
        name: 'Meridan',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #45a247, #283c86)',
        previewGradient: 'linear-gradient(to left, #45a247, #283c86)',
    },
    {
        id: 'army',
        name: 'Army',
        mode: 'dark',
        gradient: 'linear-gradient(to left, #727a17, #414d0b)',
        previewGradient: 'linear-gradient(to left, #727a17, #414d0b)',
    },
    // More light themes
    {
        id: 'lush',
        name: 'Lush',
        mode: 'light',
        gradient: 'linear-gradient(to left, #a8e063, #56ab2f)',
        previewGradient: 'linear-gradient(to left, #a8e063, #56ab2f)',
    },
    {
        id: 'mojito',
        name: 'Mojito',
        mode: 'light',
        gradient: 'linear-gradient(to left, #93F9B9, #1D976C)',
        previewGradient: 'linear-gradient(to left, #93F9B9, #1D976C)',
    },
    {
        id: 'sherbert',
        name: 'Sherbert',
        mode: 'light',
        gradient: 'linear-gradient(to left, #64f38c, #f79d00)',
        previewGradient: 'linear-gradient(to left, #64f38c, #f79d00)',
    },
    {
        id: 'emerald-water',
        name: 'Emerald Water',
        mode: 'light',
        gradient: 'linear-gradient(to left, #56B4D3, #348F50)',
        previewGradient: 'linear-gradient(to left, #56B4D3, #348F50)',
    },
    {
        id: 'easy-med',
        name: 'Easy Med',
        mode: 'light',
        gradient: 'linear-gradient(to left, #45B649, #DCE35B)',
        previewGradient: 'linear-gradient(to left, #45B649, #DCE35B)',
    },
    {
        id: 'kyoo-pal',
        name: 'Kyoo Pal',
        mode: 'light',
        gradient: 'linear-gradient(to left, #6be585, #dd3e54)',
        previewGradient: 'linear-gradient(to left, #6be585, #dd3e54)',
    },
    {
        id: 'honey-dew',
        name: 'Honey Dew',
        mode: 'light',
        gradient: 'linear-gradient(to left, #F8FFAE, #43C6AC)',
        previewGradient: 'linear-gradient(to left, #F8FFAE, #43C6AC)',
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
