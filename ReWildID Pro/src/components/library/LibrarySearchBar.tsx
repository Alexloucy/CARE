import React, { useState, useEffect } from 'react';
import { Box, TextField, InputAdornment, IconButton, Tooltip, useTheme } from '@mui/material';
import { MagnifyingGlass } from '@phosphor-icons/react';

interface LibrarySearchBarProps {
    onSearch: (query: string) => void;
}

export const LibrarySearchBar: React.FC<LibrarySearchBarProps> = ({ onSearch }) => {
    const theme = useTheme();
    const [inputValue, setInputValue] = useState('');
    const [isExpanded, setIsExpanded] = useState(false);

    // Debounce logic
    useEffect(() => {
        const timer = setTimeout(() => {
            onSearch(inputValue);
        }, 300);
        return () => clearTimeout(timer);
    }, [inputValue, onSearch]);

    return (
        <Box sx={{ 
            width: isExpanded ? '220px' : '40px', 
            transition: 'width 0.3s ease-in-out', 
            overflow: 'hidden',
            display: 'flex',
            justifyContent: 'flex-end'
        }}>
            {isExpanded ? (
                <TextField
                    autoFocus
                    placeholder="Search images..."
                    size="small"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onBlur={() => setIsExpanded(false)}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <MagnifyingGlass size={18} color={theme.palette.text.secondary} />
                            </InputAdornment>
                        ),
                        sx: {
                            borderRadius: 2,
                            bgcolor: theme.palette.background.paper,
                            width: '100%',
                            '& fieldset': { border: 'none' },
                            boxShadow: theme.palette.mode === 'dark' ? '0 0 0 1px rgba(255,255,255,0.1)' : '0 0 0 1px rgba(0,0,0,0.05)'
                        }
                    }}
                />
            ) : (
                <Tooltip title={inputValue ? `Search: ${inputValue}` : "Search"}>
                    <IconButton 
                        onClick={() => setIsExpanded(true)}
                        color={inputValue ? 'inherit' : 'default'}
                        sx={{ 
                            bgcolor: inputValue ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)') : 'transparent',
                            '&:hover': { bgcolor: inputValue ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.20)') : theme.palette.action.hover }
                        }}
                    >
                        <MagnifyingGlass weight={inputValue ? "fill" : "regular"} />
                    </IconButton>
                </Tooltip>
            )}
        </Box>
    );
};
