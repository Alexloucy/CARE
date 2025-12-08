import React, { useState, useRef, KeyboardEvent, ChangeEvent } from 'react';
import {
    Box,
    TextField,
    IconButton,
    Tooltip,
    CircularProgress,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { PaperPlaneRight, Plus, Stop } from '@phosphor-icons/react';

interface AgentInputBarProps {
    onSendMessage: (message: string) => void;
    onNewChat: () => void;
    isLoading: boolean;
    onStopGeneration?: () => void;
}

const AgentInputBar: React.FC<AgentInputBarProps> = ({
    onSendMessage,
    onNewChat,
    isLoading,
    onStopGeneration,
}) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [inputValue, setInputValue] = useState('');
    const inputRef = useRef<HTMLInputElement>(null);

    const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        setInputValue(event.target.value);
    };

    const handleSubmit = () => {
        const trimmed = inputValue.trim();
        if (trimmed && !isLoading) {
            onSendMessage(trimmed);
            setInputValue('');
        }
    };

    const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
        }
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                width: '100%',
                maxWidth: '800px',
                mx: 'auto',
                p: 2,
            }}
        >
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    width: '100%',
                    p: 1.5,
                    borderRadius: '20px',
                    background: isDark
                        ? 'rgba(255, 255, 255, 0.08)'
                        : 'rgba(0, 0, 0, 0.04)',
                    backdropFilter: 'blur(12px)',
                    WebkitBackdropFilter: 'blur(12px)',
                    border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.08)'}`,
                    boxShadow: isDark
                        ? '0 4px 20px rgba(0, 0, 0, 0.3)'
                        : '0 4px 20px rgba(0, 0, 0, 0.1)',
                }}
            >
                {/* New Chat Button */}
                <Tooltip title="New Chat">
                    <IconButton
                        onClick={onNewChat}
                        disabled={isLoading}
                        size="small"
                        sx={{
                            color: theme.palette.text.secondary,
                            '&:hover': {
                                background: isDark
                                    ? 'rgba(255, 255, 255, 0.1)'
                                    : 'rgba(0, 0, 0, 0.08)',
                            },
                        }}
                    >
                        <Plus size={20} weight="bold" />
                    </IconButton>
                </Tooltip>

                {/* Text Input */}
                <TextField
                    inputRef={inputRef}
                    fullWidth
                    variant="standard"
                    placeholder="Ask anything..."
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                    multiline
                    maxRows={4}
                    InputProps={{
                        disableUnderline: true,
                        sx: {
                            fontSize: '1rem',
                            lineHeight: 1.5,
                            px: 1,
                        },
                    }}
                    sx={{
                        flexGrow: 1,
                        '& .MuiInputBase-input': {
                            color: theme.palette.text.primary,
                            '&::placeholder': {
                                color: theme.palette.text.secondary,
                                opacity: 0.7,
                            },
                        },
                    }}
                />

                {/* Send/Stop Button */}
                {isLoading ? (
                    onStopGeneration ? (
                        <Tooltip title="Stop">
                            <IconButton
                                onClick={onStopGeneration}
                                size="small"
                                sx={{
                                    color: theme.palette.error.main,
                                    '&:hover': {
                                        background: isDark
                                            ? 'rgba(244, 67, 54, 0.2)'
                                            : 'rgba(244, 67, 54, 0.1)',
                                    },
                                }}
                            >
                                <Stop size={22} weight="fill" />
                            </IconButton>
                        </Tooltip>
                    ) : (
                        <CircularProgress size={22} sx={{ mx: 1 }} />
                    )
                ) : (
                    <Tooltip title="Send">
                        <span>
                            <IconButton
                                onClick={handleSubmit}
                                disabled={!inputValue.trim()}
                                size="small"
                                sx={{
                                    color: inputValue.trim()
                                        ? theme.palette.primary.main
                                        : theme.palette.text.disabled,
                                    '&:hover': {
                                        background: isDark
                                            ? 'rgba(255, 255, 255, 0.1)'
                                            : 'rgba(0, 0, 0, 0.08)',
                                    },
                                }}
                            >
                                <PaperPlaneRight size={22} weight="fill" />
                            </IconButton>
                        </span>
                    </Tooltip>
                )}
            </Box>
        </Box>
    );
};

export default AgentInputBar;
