import { useTheme } from '@mui/material/styles';
import { useContext, createContext } from 'react';

// Mock context for theme toggling
export const ColorModeContext = createContext({ toggleColorMode: () => { } });

export const useAppTheme = () => {
    const theme = useTheme();
    const colorMode = useContext(ColorModeContext);

    return {
        toggleTheme: colorMode.toggleColorMode,
        mode: theme.palette.mode
    };
};
