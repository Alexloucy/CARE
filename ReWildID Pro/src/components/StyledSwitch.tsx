import { Switch } from '@mui/material';
import { styled } from '@mui/material/styles';

/**
 * A custom styled switch component matching neurolink design.
 * Features a clean, flat design with proper contrast in both light and dark modes.
 */
const StyledSwitch = styled(Switch)(({ theme }) => {
    const thumbColorChecked = theme.palette.mode === 'dark'
        ? theme.palette.background.paper
        : '#fff';
    const thumbColorUnchecked = '#fff';

    return {
        width: 42,
        height: 26,
        padding: 0,
        '& .MuiSwitch-switchBase': {
            padding: 0,
            margin: 2,
            transitionDuration: '300ms',
            color: thumbColorUnchecked,
            '&.Mui-checked': {
                transform: 'translateX(16px)',
                color: thumbColorChecked,
                '& + .MuiSwitch-track': {
                    backgroundColor: theme.palette.text.primary,
                    opacity: 1,
                    border: 0,
                },
                '&.Mui-disabled + .MuiSwitch-track': {
                    opacity: 0.5,
                },
            },
            '&.Mui-focusVisible .MuiSwitch-thumb': {
                color: theme.palette.text.primary,
                border: `6px solid ${thumbColorUnchecked}`,
            },
            '&.Mui-disabled .MuiSwitch-thumb': {
                color: theme.palette.mode === 'light'
                    ? theme.palette.grey[100]
                    : theme.palette.grey[600],
            },
            '&.Mui-disabled + .MuiSwitch-track': {
                opacity: theme.palette.mode === 'light' ? 0.7 : 0.3,
            },
        },
        '& .MuiSwitch-thumb': {
            boxSizing: 'border-box',
            width: 22,
            height: 22,
            boxShadow: 'none',
        },
        '& .MuiSwitch-track': {
            borderRadius: 26 / 2,
            backgroundColor: theme.palette.mode === 'light'
                ? theme.palette.grey[300]
                : theme.palette.grey[700],
            opacity: 1,
            transition: theme.transitions.create(['background-color'], {
                duration: 500,
            }),
        },
    };
});

export default StyledSwitch;
