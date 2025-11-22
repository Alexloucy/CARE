import { PaletteMode, ThemeOptions } from '@mui/material';
import { createTheme, alpha, darken, lighten } from '@mui/material/styles';

// Define the common theme settings
const getThemeOptions = (mode: PaletteMode): ThemeOptions => ({
    palette: {
        mode,
        ...(mode === 'light'
            ? {
                // Light mode palette (Based on SnowUI Bright Light)
                primary: {
                    main: '#1c1c1c', // Bright Blue
                },
                secondary: {
                    main: '#E3F5FF', // Bright Purple
                },
                background: {
                    default: '#ffffff', // White 100%
                    paper: '#ffffff',   // White 100%
                },
                text: {
                    primary: '#000000', // Black 100%
                    secondary: 'rgba(0, 0, 0, 0.6)', // Black 80% (Approximation)
                },
                divider: 'rgba(0, 0, 0, 0.12)', // Standard light divider
            }
            : {
                // Dark mode palette (Based on SnowUI Bright Dark)
                primary: {
                    main: '#ffffff', // Brighter Blue for dark mode
                },
                secondary: {
                    main: '#BF5AF2', // Brighter Purple for dark mode
                },
                background: {
                    default: '#000000', // Black 100%
                    paper: '#1C1C1E',   // Darker paper (Similar to Black 80%)
                },
                text: {
                    primary: '#ffffff', // White 100%
                    secondary: 'rgba(255, 255, 255, 0.7)', // White 80% (Approximation)
                },
                divider: 'rgba(255, 255, 255, 0.24)', // Standard dark divider - increased visibility
            }),
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        button: {
            textTransform: 'none', // Prevent uppercase transformation
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: (theme) => `
        body {
          scrollbar-width: thin; /* For Firefox */
          scrollbar-color: ${theme.palette.mode === 'dark' ? '#555 #222' : '#aaa #eee'}; /* For Firefox */
        }
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        ::-webkit-scrollbar-track {
          background: ${theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.1) : alpha(theme.palette.background.default, 0.1)};
          border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
          background: ${theme.palette.mode === 'dark' ? alpha(theme.palette.text.secondary, 0.5) : alpha(theme.palette.text.secondary, 0.4)};
          border-radius: 10px;
          border: 2px solid transparent;
          background-clip: content-box;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: ${theme.palette.mode === 'dark' ? alpha(theme.palette.text.primary, 0.7) : alpha(theme.palette.text.primary, 0.6)};
        }
        ::-webkit-scrollbar-corner {
          background: transparent;
        }
      `,
        },
        MuiAppBar: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: ({ theme }) => ({
                    boxShadow: theme.palette.mode === 'light'
                        ? '0 2px 4px rgba(0,0,0,0.1)'
                        : '0 2px 4px rgba(0,0,0,0.3)',
                }),
            },
        },
        MuiCard: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: ({ theme }) => {
                    const cardBgLight = '#F7F9FB';
                    const cardBgDark = '#1e1e24';

                    return {
                        borderRadius: '16px',
                        border: 'none',
                        boxShadow: 'none',
                        backgroundColor: theme.palette.mode === 'light'
                            ? cardBgLight
                            : cardBgDark,
                        minHeight: 280,
                        height: '100%',
                        width: '100%',
                        transition: theme.transitions.create('background-color', {
                            duration: theme.transitions.duration.short,
                        }),
                        '&:hover': {
                            backgroundColor: theme.palette.mode === 'light'
                                ? darken(cardBgLight, 0.01)
                                : lighten(cardBgDark, 0.04),
                        },
                    };
                },
            },
        },
        MuiButton: {
            defaultProps: {
                disableElevation: true,
            },
            styleOverrides: {
                root: {
                    borderRadius: '8px',
                    textTransform: 'none',
                    fontWeight: 600,
                    padding: '8px 16px',
                },
                containedPrimary: ({ theme }) => ({
                    backgroundColor: theme.palette.mode === 'light'
                        ? '#1C1C1C'
                        : '#ffffff',
                    color: theme.palette.mode === 'light'
                        ? '#ffffff'
                        : '#000000',
                    '&:hover': {
                        backgroundColor: theme.palette.mode === 'light'
                            ? '#3C3C3C'
                            : '#f0f0f0',
                    },
                }),
            },
        },
    },
});

export const lightTheme = createTheme(getThemeOptions('light'));
export const darkTheme = createTheme(getThemeOptions('dark'));

export const getTheme = (mode: PaletteMode) =>
    mode === 'light' ? lightTheme : darkTheme;
